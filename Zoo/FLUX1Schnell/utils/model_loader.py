import gc
import json
import os
from typing import List, Optional
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def get_safetensors_files(repo_id: str, subfolder: str) -> List[str]:
    """
    Locates and downloads safetensors shards without loading them into memory.
    """
    # 1. Check for single file
    for single_name in ["diffusion_pytorch_model.safetensors", "model.safetensors"]:
        try:
            return [hf_hub_download(repo_id=repo_id, filename=single_name, subfolder=subfolder)]
        except Exception:
            pass

    # 2. Check for index JSON
    for index_name in ["diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"]:
        try:
            index_path = hf_hub_download(repo_id=repo_id, filename=index_name, subfolder=subfolder)
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            shard_filenames = sorted(list(set(data["weight_map"].values())))
            return [
                hf_hub_download(repo_id=repo_id, filename=fname, subfolder=subfolder)
                for fname in shard_filenames
            ]
        except Exception:
            pass

    # 3. Fallback discovery
    found = []
    for i in range(1, 15):
        for total in [2, 3, 4, 5]:
            for prefix in ["diffusion_pytorch_model", "model"]:
                fname = f"{prefix}-{i:05d}-of-{total:05d}.safetensors"
                try:
                    p = hf_hub_download(repo_id=repo_id, filename=fname, subfolder=subfolder)
                    found.append(p)
                    break
                except Exception:
                    pass
    if found:
        return sorted(list(set(found)))

    raise FileNotFoundError(f"Could not locate safetensors weights for {subfolder} in {repo_id}")


def _set_module_tensor(root_module: nn.Module, target_key: str, tensor: torch.Tensor):
    """Navigates and replaces a module parameter or buffer in-place."""
    parts = target_key.split(".")
    cur = root_module
    for part in parts[:-1]:
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, (nn.ModuleList, nn.Sequential, list)) and part.isdigit():
            cur = cur[int(part)]
        else:
            return False

    leaf = parts[-1]
    if leaf in cur._parameters:
        cur._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in cur._buffers:
        cur._buffers[leaf] = tensor
    else:
        setattr(cur, leaf, nn.Parameter(tensor, requires_grad=False))
    return True


def stream_safetensors_to_model(
    model: nn.Module,
    file_paths: List[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    key_prefix_strip: Optional[str] = None,
):
    """
    Streams tensors directly from disk into GPU memory one by one.
    Peak System RAM overhead: ~size of a single tensor (< 100 MB).
    """
    for file_path in file_paths:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                mapped_key = key
                if key_prefix_strip and mapped_key.startswith(key_prefix_strip):
                    mapped_key = mapped_key[len(key_prefix_strip) :]

                # Read only this specific tensor from disk and immediately transfer to GPU
                tensor = f.get_tensor(key).to(device=device, dtype=dtype)
                success = _set_module_tensor(model, mapped_key, tensor)

                # Special case for T5 token embedding weight tying
                if mapped_key == "shared.weight":
                    _set_module_tensor(model, "encoder.embed_tokens.weight", tensor)
                elif mapped_key == "encoder.embed_tokens.weight":
                    _set_module_tensor(model, "shared.weight", tensor)

                del tensor

    # Materialize any remaining meta parameters (e.g. omitted biases) as zeros on target device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            zero_param = torch.zeros(param.shape, device=device, dtype=dtype)
            _set_module_tensor(model, name, zero_param)

    # Materialize any remaining meta buffers
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            zero_buf = torch.zeros(buffer.shape, device=device, dtype=dtype)
            _set_module_tensor(model, name, zero_buf)