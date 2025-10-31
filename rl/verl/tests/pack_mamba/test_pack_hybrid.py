from __future__ import annotations

"""Mamba packed‑vs‑unpacked inference regression test.

Run as a script or import the ``test_mamba_block`` function from elsewhere.

Example::

    python mamba_block_tester.py --seq-len 8192 --packages 8 --dtype bf16
"""

import argparse
import copy
import time
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch
from verl.models.mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper


################################################################################
# Utilities
################################################################################


@contextmanager
def align_timer(label: str = "no_name"):
    """Measure elapsed time (ms) for a single CUDA kernel or CPU block.

    Automatically synchronizes CUDA so that *only* the targeted code is timed.
    Falls back to wall‑clock timing if CUDA is unavailable.
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        start_time = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start_time) * 1_000

    print(f"{label:15s}: {elapsed_ms:8.3f} ms")


def calc_tolerance(dtype: torch.dtype) -> Tuple[float, float]:
    """Return rtol/atol tuned for *approximate* equality on given dtype."""

    if dtype == torch.float32:
        return 6e-4, 2e-3
    if dtype == torch.bfloat16:
        return 3e-2, 5e-2
    # default for fp16 or others
    return 3e-3, 5e-3


def build_position_ids(
    seq_len: int, packages: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Return contiguous position‑ids tensor for equal segmentation."""

    seg_len = seq_len // packages
    return (
        torch.arange(seg_len, device=device, dtype=torch.int32)
        .repeat(packages)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )


################################################################################
# Core logic
################################################################################


@torch.inference_mode()
def forward_compare(
    model: MambaTransformerHybridModelWrapper,
    model_ref: MambaTransformerHybridModelWrapper,
    packed_input: torch.Tensor,
    unpacked_input: torch.Tensor,
    position_ids: torch.Tensor | None,
    rtol: float,
    atol: float,
) -> None:
    """Run packed/unpacked forward passes and assert numeric equivalence."""

    batch_size, total_tokens = packed_input.shape
    packages = unpacked_input.size(0) // batch_size
    seg_len = unpacked_input.size(1)

    with align_timer("packed fwd"):
        out = model(packed_input, position_ids=position_ids).logits
    with align_timer("unpacked fwd"):
        out_ref = model_ref(unpacked_input).logits

    repacked = (
        out_ref.view(batch_size, packages, seg_len, -1)
        .reshape(batch_size, total_tokens, -1)
    )

    diff = (out - repacked).abs()
    print(
        f"max diff {diff.max().item():.4e}   mean diff {diff.mean().item():.4e}"
    )

    np.testing.assert_allclose(
        out.float().cpu().numpy(),
        repacked.float().cpu().numpy(),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(out, repacked, rtol=rtol, atol=atol)


def test_mamba_block(
    *,
    seq_len: int = 4_096,
    batch_size: int = 1,
    packages: int = 4,
    vocab_size: int = 32_000,
    model_name: str = "JunxiongWang/M1-3B",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Compare logits between packed and unpacked inference for a Mamba model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure seq_len divisible by packages; pad upward if needed
    if seq_len % packages:
        seq_len = ((seq_len + packages - 1) // packages) * packages
        print(f"[info] seq_len padded to {seq_len} for equal segmentation.")

    seg_len = seq_len // packages
    total_tokens = seg_len * packages

    # Synthetic input ids
    rng = torch.Generator(device=device).manual_seed(0)
    packed = torch.randint(
        vocab_size, (batch_size, total_tokens), generator=rng, device=device
    )
    unpacked = (
        packed.view(batch_size, packages, seg_len)
        .reshape(batch_size * packages, seg_len)
    )
    pos_ids = build_position_ids(seq_len, packages, batch_size, device)

    # Load model(s)
    with align_timer("load model"):
        model = MambaTransformerHybridModelWrapper.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
    model_ref = copy.deepcopy(model).eval()

    rtol, atol = calc_tolerance(dtype)
    forward_compare(model, model_ref, packed, unpacked, pos_ids, rtol, atol)


################################################################################
# CLI
################################################################################


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mamba block tester")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--packages", type=int, default=4)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--model", default="JunxiongWang/M1-3B", help="HuggingFace model ID"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    test_mamba_block(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        packages=args.packages,
        dtype=dtype,
        model_name=args.model,
    )


if __name__ == "__main__":  # pragma: no cover
    main()