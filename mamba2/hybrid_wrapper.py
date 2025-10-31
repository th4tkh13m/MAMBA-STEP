
# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from mamba2._generation import GenerationMixin

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf
from transformers.utils.hub import cached_file

from mamba2.hybrid_mamba_config import MambaConfig
from mamba2.hybrid_model import MambaDecoderLayer, MHADecoderLayer
from mamba2.util import load_state_dict_hf, load_safetensors_to_dict
from collections import namedtuple


def prepare_fa2_from_position_ids(position_ids):
    position_flatten_ids = position_ids.flatten()
    indices_q = torch.arange(position_flatten_ids.size(0), device=position_flatten_ids.device, dtype=torch.int32)
    cu_seq_lens = torch.cat(
        (
            indices_q[position_flatten_ids == 0],
            torch.tensor(position_flatten_ids.size(), device=position_flatten_ids.device, dtype=torch.int32),
        )
    )
    max_length = (position_flatten_ids.max() + 1).item()
    return cu_seq_lens, max_length


def merge_projections_for_layers(checkpoint, layer_indices):
    for layer_idx in layer_indices:
        # Get the weights for q_proj, k_proj, and v_proj
        q_proj_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_proj_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_proj_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        o_proj_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"

        # Check if the keys exist in the checkpoint
        if q_proj_key in checkpoint and k_proj_key in checkpoint and v_proj_key in checkpoint:
            # Assuming all the projections have the same shape, otherwise adjust accordingly
            q_proj_weight = checkpoint[q_proj_key]
            k_proj_weight = checkpoint[k_proj_key]
            v_proj_weight = checkpoint[v_proj_key]

            # Concatenate the weights along the first dimension (often dimension 0)
            in_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)

            # Assign the new weight to the corresponding in_proj key
            in_proj_key = f"model.layers.{layer_idx}.mha.in_proj.weight"
            checkpoint[in_proj_key] = in_proj_weight

            # Optionally, remove the old keys to clean up the checkpoint
            del checkpoint[q_proj_key]
            del checkpoint[k_proj_key]
            del checkpoint[v_proj_key]

        if o_proj_key in checkpoint:
            out_proj_key = f"model.layers.{layer_idx}.mha.out_proj.weight"
            checkpoint[out_proj_key] = checkpoint[o_proj_key]
            del checkpoint[o_proj_key]

    return checkpoint


def unpad_input_ids(input_ids: torch.LongTensor, attention_mask: torch.Tensor):
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    seq_lengths = attention_mask.sum(dim=1).to(torch.int32)
    max_seqlen = int(seq_lengths.max().item())
    cu_seqlens = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lengths, dim=0)
    unpad_input_ids = input_ids[attention_mask]                     # (total_tokens,)
    shifted = attention_mask.cumsum(dim=1).to(torch.int32) - 1      # shape: (bsz, seq_len)
    position_ids = shifted[attention_mask]                          # shape: (total_tokens,)
    return unpad_input_ids.unsqueeze(0), position_ids.unsqueeze(0), cu_seqlens, max_seqlen


def get_last_n_states(hidden: torch.Tensor, cu_seqlens: torch.LongTensor, num_last_tokens: int):
    hidden_flat = hidden.squeeze(0)
    end_indices = cu_seqlens[1:] - 1
    offsets = torch.arange(-num_last_tokens + 1, 1, device=hidden.device)
    last_indices = end_indices.unsqueeze(1) + offsets.unsqueeze(0)
    return hidden_flat[last_indices]


MAMBA_CONFIG_NAME = "mamba_config.json"

class MambaTransformerHybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, load_from_hub=False, **kwargs):
        super(MambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.ssm_layers = [i for i in range(mamba_config.n_layer) if i not in attn_layers]
        self.model = transformer_model
        self.config = self.model.config
        
        for layer_idx in range(mamba_config.n_layer):
            if layer_idx in attn_layers:
                layer_encoder = MHADecoderLayer(
                    self.config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            else:
                layer_encoder = MambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            self.model.model.layers[layer_idx] = layer_encoder
            
        print("self.model:", self.model)      
           
        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                ckpt = load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype)
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    ckpt = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu"))
                else:
                    # support save from safetensors
                    ckpt = load_safetensors_to_dict(checkpoint_path)
        
            merge_projections_for_layers(ckpt, self.attn_layers)
            self.model.load_state_dict(ckpt)

        self.model = self.model.to(dtype).cuda()
        self.device = self.model.device
        self.dtype = dtype

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if inference_params is None and num_last_tokens == 0:
            # we want to reuse hf features, so call hf forward to support gradient checkpoints
            # this is happen in the training
            attention_mask = mixer_kwargs.pop("attention_mask", None)
            if position_ids is not None:
                if position_ids.dtype == torch.int64:
                    position_ids = position_ids.to(torch.int32)

                cu_seqlens, max_seqlen = prepare_fa2_from_position_ids(position_ids)
                mixer_kwargs.update({
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen,
                })

            return self.model(input_ids, attention_mask, position_ids, **mixer_kwargs)
        else:

            cu_seqlens = None
            if inference_params.seqlen_offset == 0:
                # prefill
                attention_mask = (input_ids != self.pad_token_id)
                input_ids, position_ids, cu_seqlens, max_seqlen = unpad_input_ids(input_ids, attention_mask)
                seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=cu_seqlens.device) 
                        for i, s in enumerate(cu_seqlens[1:]-cu_seqlens[:-1])], dim=0).unsqueeze(0)
                mixer_kwargs.update({
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen,
                    "seq_idx": seq_idx,
                })

            # this if for inference
            hidden_states = self.model.model.embed_tokens(input_ids)

            for layer_id, decoder_layer in enumerate(self.model.model.layers):
                hidden_states = decoder_layer(
                    hidden_states, position_ids=position_ids, inference_params=inference_params, **mixer_kwargs
                )
            
            hidden_states = self.model.model.norm(hidden_states)
            if inference_params.seqlen_offset == 0 and cu_seqlens is not None:
                # variable length 
                # inference, prefill
                hidden_states = get_last_n_states(hidden_states, cu_seqlens, num_last_tokens)
                if inference_params.lengths_per_sample is None:
                    inference_params.lengths_per_sample = torch.full(
                        (cu_seqlens.shape[0] - 1,), 0, dtype=torch.int32, device=cu_seqlens.device
                    )
                inference_params.lengths_per_sample += attention_mask.sum(-1)
            else:
                # regular default path
                if num_last_tokens > 0:
                    hidden_states = hidden_states[:, -num_last_tokens:]

                if inference_params.lengths_per_sample is None:
                    inference_params.lengths_per_sample = torch.full(
                        (input_ids.shape[0], ), input_ids.shape[1], dtype=torch.int32, device=input_ids.device
                )

            if inference_params.seqlen_offset > 0:
                inference_params.lengths_per_sample += 1

            lm_logits = self.model.lm_head(hidden_states)

            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=lm_logits)

    @staticmethod
    def from_pretrained_local(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            config_data["_name_or_path"],
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        with open(f"{pretrained_model_name}/{MAMBA_CONFIG_NAME}", "r") as json_file:
            config_dict = json.load(json_file)
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(
            pretrained_model_name,
            transformer_model,
            mamba_config,
            mamba_config.attn_layers,
            torch_dtype,
            init_with_kqvo=False,
        )

    @staticmethod
    def from_pretrained_hub(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            config_data["_name_or_path"],
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        resolved_archive_file = cached_file(
            pretrained_model_name,
            MAMBA_CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        config_dict = json.load(open(resolved_archive_file))
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(
            pretrained_model_name,
            transformer_model,
            mamba_config,
            mamba_config.attn_layers,
            torch_dtype,
            init_with_kqvo=False,
            load_from_hub=True,
        )

    @staticmethod
    def from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        if os.path.exists(pretrained_model_name):
            return MambaTransformerHybridModelWrapper.from_pretrained_local(
                pretrained_model_name, torch_dtype, attn_implementation
            )
        else:
            return MambaTransformerHybridModelWrapper.from_pretrained_hub(
                pretrained_model_name, torch_dtype, attn_implementation
            )

    def save_pretrained(self, save_directory, state_dict):
        # Ensure save_directory exists
        self.save_config(save_directory)
        self.model.save_pretrained(save_directory, state_dict=state_dict)

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "mamba_config.json")
        with open(config_path, "w") as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)

    def get_memory_footprint(self):
        return self.model.get_memory_footprint()

    def generate(
        self,
        input_ids,
        max_length=1024,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        if kwargs is not None:

            max_new_tokens = kwargs.pop("max_new_tokens", None)
            if max_new_tokens is not None:
                max_length = max_new_tokens + input_ids.shape[1]

            cg = kwargs.pop("cg", True)
            eos_token_id = kwargs.pop("eos_token_id", None)
            self.pad_token_id = kwargs.pop("pad_token_id", None)

        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=cg,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            eos_token_id=eos_token_id,
            **kwargs,
        )
