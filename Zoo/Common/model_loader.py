"""Universal Hugging Face Model & Weight Loader with streaming shard loading,

auto-dtype detection, and global strict key enforcement.
"""

import gc
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file


def auto_detect_device_and_dtype(
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[str, torch.dtype]:
    """Automatically detects the optimal device and precision format.

    - CUDA with bf16 support (Ampere/Ada/Hopper): torch.bfloat16
    - CUDA without bf16 (Turing/Pascal e.g. T4): torch.float16
    - Apple Silicon MPS: torch.float16
    - CPU: torch.float32 (for numerical stability and compatibility)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_str = str(device)

    if dtype is None:
        if "cuda" in device_str:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif "mps" in device_str:
            dtype = torch.float16
        else:
            dtype = torch.float32

    return device_str, dtype


def load_hf_model_weights(
    model: torch.nn.Module,
    repo_id: str,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    strict: bool = True,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Path:
    """Downloads weights and streams shards directly into the model to avoid RAM OOM.

    Args:
        model: PyTorch model module to populate.
        repo_id: Hugging Face repository identifier.
        allow_patterns: Glob patterns for checkpoint downloading.
        strict: Enforce exact key matching across all shards globally.
        device: Target device (auto-detected if None).
        dtype: Target precision dtype (auto-detected if None).

    Returns:
        Path: Path to the cached model directory.
    """
    resolved_device, resolved_dtype = auto_detect_device_and_dtype(device, dtype)

    if allow_patterns is None:
        allow_patterns = ["*.safetensors", "*.pt", "*.bin", "*.json"]
    if ignore_patterns is None:
        # Exclude Mistral's raw single-file consolidated weights to save 50% download/disk
        ignore_patterns = ["*consolidated*.safetensors", "consolidated.safetensors"]

    print(f"Fetching weights for '{repo_id}' from Hugging Face Cache...")
    cache_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        )
    )

    # 1. Determine exact safetensors shard files to load
    index_file = cache_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_names = sorted(list(set(index_data.get("weight_map", {}).values())))
        safetensors_files = [cache_dir / name for name in shard_names if (cache_dir / name).exists()]
    else:
        # Fallback globbing, strictly ignoring consolidated weights
        safetensors_files = sorted(
            [p for p in cache_dir.glob("*.safetensors") if "consolidated" not in p.name.lower()]
        )

    pt_files = sorted(list(cache_dir.glob("*.pt"))) + sorted(list(cache_dir.glob("*.bin")))

    # Convert model parameters to target dtype first
    model.to(dtype=resolved_dtype)

    # Track keys across all shards globally
    all_model_keys = set(model.state_dict().keys())
    missing_keys = set(all_model_keys)
    unexpected_keys = set()

    print(f"Loading weights (Target: {resolved_device}, {resolved_dtype}, strict={strict})...")

    # 2. Safetensors streaming: process one shard at a time
    if safetensors_files:
        for idx, shard_file in enumerate(safetensors_files):
            print(f"  Streaming shard {idx + 1}/{len(safetensors_files)}: {shard_file.name}...")
            shard_dict = load_file(str(shard_file), device="cpu")
            
            # Discard serialized position_ids so persistent=False never fails strict check ---
            for k in list(shard_dict.keys()):
                if k.endswith("position_ids"):
                    del shard_dict[k]

            shard_keys = set(shard_dict.keys())
            unexpected_keys.update(shard_keys - all_model_keys)
            missing_keys.difference_update(shard_keys)

            converted_shard = {k: v.to(dtype=resolved_dtype) for k, v in shard_dict.items()}
            del shard_dict

            model.load_state_dict(converted_shard, strict=False)

            del converted_shard
            gc.collect()

    # 3. PyTorch binary fallback
    elif pt_files:
        for idx, shard_file in enumerate(pt_files):
            print(f"  Loading binary checkpoint: {shard_file.name}...")
            loaded = torch.load(shard_file, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict) and "model" in loaded:
                loaded = loaded["model"]
                
            for k in list(loaded.keys()):
                if k.endswith("position_ids"):
                    del loaded[k]

            shard_keys = set(loaded.keys())
            unexpected_keys.update(shard_keys - all_model_keys)
            missing_keys.difference_update(shard_keys)

            converted = {k: v.to(dtype=resolved_dtype) for k, v in loaded.items()}
            del loaded

            model.load_state_dict(converted, strict=False)

            del converted
            gc.collect()
    else:
        raise FileNotFoundError(f"No valid checkpoint shards found in {cache_dir}")

    # 4. Automatically tie weights if model supports it (resolves omitted lm_head.weight)
    if hasattr(model, "tie_weights"):
        tie_weights_fn = getattr(model, "tie_weights", None)
        if callable(tie_weights_fn):
            tie_weights_fn()
            # Remove tied keys from missing_keys since they are now bound to embed_tokens
            missing_keys.discard("language_model.lm_head.weight")
            missing_keys.discard("lm_head.weight")

    # 5. Enforce global strictness across all shards
    if strict:
        error_msgs = []
        if unexpected_keys:
            error_msgs.append(f"Unexpected key(s) in state_dict: {sorted(list(unexpected_keys))}")
        if missing_keys:
            error_msgs.append(f"Missing key(s) in state_dict: {sorted(list(missing_keys))}")

        if error_msgs:
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__} (strict=True):\n\t"
                + "\n\t".join(error_msgs)
            )
    else:
        if unexpected_keys:
            print(f"  Unexpected keys ({len(unexpected_keys)}): {sorted(list(unexpected_keys))[:5]}...")
        if missing_keys:
            print(f"  Missing keys ({len(missing_keys)}): {sorted(list(missing_keys))[:5]}...")

    model.to(device=resolved_device)
    model.eval()
    print(f"✓ Model successfully loaded onto {resolved_device.upper()} in {resolved_dtype}.")
    return cache_dir


def purge_memory() -> None:
    """Aggressively purges system RAM and GPU VRAM cached memory blocks.

    Invokes Python garbage collection, followed by backend-specific cache
    deallocation (CUDA cache and IPC collection, or Apple Silicon MPS cache).
    Essential between stages in multi-gigabyte sequential pipelines (e.g. FLUX).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_submodule_tensor(module: torch.nn.Module, subkey: str, tensor: torch.Tensor) -> None:
    """Copies tensor data directly into an existing submodule parameter in-place.

    Traverses hierarchical parameter names (including numeric indices for ModuleList
    and Sequential containers) and updates the target parameter's underlying buffer
    without reallocating parameter objects.

    Args:
        module: Root PyTorch module containing the target parameter.
        subkey: Dot-separated relative key path (e.g., 'transformer_blocks.0.attn.to_q.weight').
        tensor: Source tensor whose values will be copied into the destination parameter.

    Raises:
        AttributeError: If any intermediate submodule or leaf parameter cannot be found.
    """
    parts = subkey.split(".")
    curr: Any = module
    for part in parts[:-1]:
        if part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)
    leaf = parts[-1]
    param = getattr(curr, leaf)
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def get_safetensors_shards(repo_id: str, subfolder: str = "") -> List[str]:
    """Discovers and downloads all single or sharded safetensors files for a model subfolder.

    First checks for standard monolithic weights, then attempts to parse the index
    JSON weight map, and finally falls back to sequential shard naming discovery.
    Downloads files to the local cache without loading tensor bytes into system RAM.

    Args:
        repo_id: Hugging Face repository identifier (e.g., 'black-forest-labs/FLUX.1-schnell').
        subfolder: Subdirectory inside the repository (e.g., 'transformer', 'text_encoder_2').

    Returns:
        Sorted list of absolute local filesystem paths to downloaded safetensors shards.

    Raises:
        FileNotFoundError: If no matching safetensors checkpoint files can be located.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    # 1. Single-file check
    for single_name in ["diffusion_pytorch_model.safetensors", "model.safetensors"]:
        try:
            return [hf_hub_download(repo_id=repo_id, filename=single_name, subfolder=subfolder, token=token)]
        except Exception:
            pass

    # 2. Sharded index JSON check
    for index_name in ["diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"]:
        try:
            index_path = hf_hub_download(repo_id=repo_id, filename=index_name, subfolder=subfolder, token=token)
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            shard_filenames = sorted(list(set(data["weight_map"].values())))
            return [
                hf_hub_download(repo_id=repo_id, filename=fname, subfolder=subfolder, token=token)
                for fname in shard_filenames
            ]
        except Exception:
            pass

    # 3. Fallback sequential pattern discovery (up to 15 shards)
    found_paths: List[str] = []
    for i in range(1, 16):
        shard_found = False
        for prefix in ["diffusion_pytorch_model", "model"]:
            for total in [2, 3, 4, 5, 6, 7, 8]:
                fname = f"{prefix}-{i:05d}-of-{total:05d}.safetensors"
                try:
                    p = hf_hub_download(repo_id=repo_id, filename=fname, subfolder=subfolder, token=token)
                    found_paths.append(p)
                    shard_found = True
                    break
                except Exception:
                    continue
            if shard_found:
                break
        if not shard_found and found_paths:
            break

    if not found_paths:
        raise FileNotFoundError(f"Could not locate safetensors weights for '{subfolder}' in '{repo_id}'")
    return sorted(list(set(found_paths)))