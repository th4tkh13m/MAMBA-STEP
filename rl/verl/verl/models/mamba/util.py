import os
import torch
import json

from safetensors.torch import load_file
from safetensors import safe_open
from transformers.utils import cached_file, is_safetensors_available, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, is_safetensors_available

def load_bins_to_dict(directory):
    safetensors_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.bin'):
            print(filename)
            file_path = os.path.join(directory, filename)
            f = torch.load(file_path, map_location=torch.device("cpu"))
            for key, tensor in f.items():
                safetensors_dict[key] = tensor
    return safetensors_dict

def load_safetensors_to_dict(directory):
    safetensors_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    safetensors_dict[key] = f.get_tensor(key)
    return safetensors_dict

def construct_layer_dict(safetensors_dict, num_hidden_layers):
    layer_dict = {}
    is_mamba_layer = [False for _ in range(num_hidden_layers)]
    prefix = "model.layers."
    for full_key, tensor in safetensors_dict.items():
        if full_key.startswith(prefix):
            parts = full_key[len(prefix):].split('.', 1)
            layer_id = int(parts[0])
            param_name = parts[1]
            if layer_id not in layer_dict:
                layer_dict[layer_id] = {}
            if "mamba" in param_name:
                is_mamba_layer[layer_id] = True
            layer_dict[layer_id][param_name] = tensor
    return layer_dict, is_mamba_layer

def load_state_dict_hf(model_name, device=None, dtype=None):
    # Determine the appropriate device for loading
    mapped_device = "cpu"

    # Check if safetensors is available
    if is_safetensors_available():
        # Attempt to load the index file for sharded safetensors
        try:
            index_file = cached_file(model_name, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False)
        except EnvironmentError:
            index_file = None

        if index_file:
            # Load the index file to get the list of shard filenames
            with open(index_file, "r") as f:
                index_data = json.load(f)
                shard_filenames = list(set(index_data["weight_map"].values()))

            # Initialize an empty state dictionary
            state_dict = {}

            # Load each shard and update the state dictionary
            for shard_filename in shard_filenames:
                shard_path = cached_file(model_name, shard_filename)
                shard_state_dict = load_file(shard_path, device=mapped_device)
                state_dict.update(shard_state_dict)

            return state_dict
        else:
            # Attempt to load a single safetensors file
            try:
                safetensors_file = cached_file(model_name, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            except EnvironmentError:
                safetensors_file = None

            if safetensors_file:
                return load_file(safetensors_file, device=mapped_device)

    # Fallback to a single .bin file
    try:
        try:
            index_file = cached_file(model_name, WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False)
        except EnvironmentError:
            index_file = None

        if index_file:
            # Load the index file to get the list of shard filenames
            with open(index_file, "r") as f:
                index_data = json.load(f)
                shard_filenames = list(set(index_data["weight_map"].values()))

            # Initialize an empty state dictionary
            state_dict = {}

            # Load each shard and update the state dictionary
            for shard_filename in shard_filenames:
                shard_path = cached_file(model_name, shard_filename)
                shard_state_dict = load_file(shard_path, device=mapped_device)
                state_dict.update(shard_state_dict)

            return state_dict
        else:
            bin_file = cached_file(model_name, WEIGHTS_NAME)

    except EnvironmentError:
        raise FileNotFoundError(f"No model weights found for {model_name} in .safetensors or .bin format.")

    return torch.load(bin_file, map_location=mapped_device)