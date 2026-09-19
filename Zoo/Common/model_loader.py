"""Universal Hugging Face Model & Weight Loader with streaming shard loading,

auto-dtype detection, and global strict key enforcement.
"""

import gc
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from huggingface_hub import snapshot_download
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