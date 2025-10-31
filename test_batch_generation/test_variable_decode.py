import torch
from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba._generation import InferenceParams
from transformers import AutoTokenizer

import random

seqlens = [random.randint(10, 1000) for _ in range(100)]
max_new = 100

device = torch.device("cuda")

xs = [torch.randint(1, 128000, (1, seqlen), device="cuda", dtype=torch.long) for seqlen in seqlens]

tensor_type = torch.float16

model_name = "togethercomputer/M1-3B"
model = MambaTransformerHybridModelWrapper.from_pretrained(
    model_name, torch_dtype=tensor_type
).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model.pad_token_id = tokenizer.pad_token_id
print(tokenizer.pad_token_id)

with torch.no_grad():

    batch_size = len(seqlens)
    max_len_input = max(seqlens)

    cur_lens = torch.tensor(seqlens, device=device)
    total_len = max_len_input + max_new

    kv_cache = model.allocate_inference_cache(batch_size, total_len, dtype=tensor_type)
    infp = InferenceParams(
        max_seqlen=total_len,
        max_batch_size=batch_size,
        seqlen_offset=0,
        key_value_memory_dict=kv_cache,
        lengths_per_sample=torch.zeros(batch_size, dtype=torch.int32, device=device),
    )

    padded_input = torch.full((batch_size, max_len_input), tokenizer.pad_token_id, dtype=torch.long, device=device)  # padding with token 0

    for i, x in enumerate(xs):
        padded_input[i, -x.shape[1]:] = x

    logits = model(padded_input, position_ids=None, inference_params=infp, num_last_tokens=1).logits.squeeze(1)
    next_tok = logits.argmax(-1)
    generated_batch = [next_tok.clone().unsqueeze(1)]

    infp.seqlen_offset += max_len_input

    print("infp.lengths_per_sample:", infp.lengths_per_sample)
    
    all_logits2 = [logits]

    for _ in range(max_new):
        next_tok2 = next_tok
        x_step = next_tok.unsqueeze(1)
        logits = model(x_step, position_ids=None, inference_params=infp, num_last_tokens=1).logits.squeeze(1)
        next_tok = logits.argmax(-1)
        generated_batch.append(next_tok.clone().unsqueeze(1))
        infp.seqlen_offset += 1
        all_logits2.append(logits)

    all_logits1 =  []
    per_chunk_caches = [] 
    next_tok1 = []
    for i, toks in enumerate(xs):
        all_logits = []
        B, cur_len = toks.shape
        max_len = cur_len + max_new

        kv_cache_ref = model.allocate_inference_cache(B, max_len, dtype=tensor_type)
        infp_ref = InferenceParams(
            max_seqlen        = max_len,
            max_batch_size    = B,
            seqlen_offset     = 0,
            key_value_memory_dict = kv_cache_ref,
        )

        logits = model(toks, position_ids=None,
                        inference_params=infp_ref, num_last_tokens=1).logits.squeeze(1)

        # Use the same tokens as the packed version
        next_tok = logits.argmax(-1)
        generated = [generated_batch[0][i].clone()]

        next_tok1.append(next_tok)

        infp_ref.seqlen_offset += cur_len
        all_logits.append(logits)

        for j in range(max_new):
            x = generated_batch[j][i].unsqueeze(0)  # (B_alive,1)
            logits = model(x, position_ids=None,
                            inference_params=infp_ref, num_last_tokens=1).logits.squeeze(1)

            # Use the same tokens as the packed version
            generated.append(generated_batch[j+1][i].clone())

            infp_ref.seqlen_offset += 1
            all_logits.append(logits)

        all_logits = torch.stack(all_logits, dim=1)
        all_logits1.append(all_logits)
        per_chunk_caches.append(kv_cache_ref)

    all_logits1 = torch.concat(all_logits1, dim=0)  # (B, T, V)
    all_logits2 = torch.stack(all_logits2, dim=1)  # (B, T, V)

    print("prefill max:", torch.max(torch.abs(all_logits1[:, 0, ...] - all_logits2[:, 0, ...])))
    print("final max:", torch.max(torch.abs(all_logits1 - all_logits2)))

    print("prefill mean:", torch.mean(torch.abs(all_logits1[:, 0, ...] - all_logits2[:, 0, ...])))
    print("final mean:", torch.mean(torch.abs(all_logits1 - all_logits2)))

    print("=======================")

    max_ssm_diff = 0
    max_conv_diff = 0

    max_conv_diff1 = 0
    max_conv_diff2 = 0
    max_conv_diff3 = 0
    max_conv_diff4 = 0

    for layer_id in model.ssm_layers:
        
        pack_conv_cache, pack_ssm_cache = infp.key_value_memory_dict[layer_id]

        for i in range(len(seqlens)):
            # In the "packed" run,ach chunk's state is at index i along dimension 0
            conv_i_pack = pack_conv_cache[i]   # shape depends on Mamba's internals, e.g. [batch_size, ...]
            ssm_i_pack  = pack_ssm_cache[i]

            # In the reference run, chunk i's state is in per_chunk_caches[i]
            conv_i_ref, ssm_i_ref = per_chunk_caches[i][layer_id]

            # print(conv_i_pack.shape, conv_i_ref.shape)
            conv_diff = (conv_i_pack - conv_i_ref).abs()
            ssm_diff  = (ssm_i_pack  - ssm_i_ref).abs()

            max_conv_diff1 = max(max_conv_diff1, conv_diff[..., 0].max().item())
            max_conv_diff2 = max(max_conv_diff2, conv_diff[..., 1].max().item())
            max_conv_diff3 = max(max_conv_diff3, conv_diff[..., 2].max().item())
            max_conv_diff4 = max(max_conv_diff4, conv_diff[..., 3].max().item())

            max_ssm_diff = max(max_ssm_diff, ssm_diff.max().item())
            max_conv_diff = max(max_conv_diff, conv_diff.max().item())  
    
    print(f"Max SSM cache difference: {max_ssm_diff:.6f}")
    print(f"Max Conv cache difference: {max_conv_diff:.6f}")

    print(f"Max Conv cache difference 1: {max_conv_diff1:.6f}")
    print(f"Max Conv cache difference 2: {max_conv_diff2:.6f}")
    print(f"Max Conv cache difference 3: {max_conv_diff3:.6f}")
    print(f"Max Conv cache difference 4: {max_conv_diff4:.6f}")