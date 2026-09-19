"""Universal Hugging Face Model & Weight Loader using native HF Cache."""

from pathlib import Path
from typing import List, Optional
import os
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_hf_model_weights(
    model: torch.nn.Module,
    repo_id: str,
    allow_patterns: Optional[List[str]] = None,
    strict: bool = True,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Path:
    """Downloads weights to standard HF cache and loads them directly into a PyTorch model.

    Note: Omitting local_dir ensures weights stay in ~/.cache/huggingface/hub to prevent
    duplicate local disk consumption.

    Returns:
        Path: Path to the cached directory (useful for loading tokenizers / configs).
    """
    if allow_patterns is None:
        allow_patterns = ["*.safetensors", "*.pt", "*.bin", "*.json"]

    print(f"Fetching weights for '{repo_id}' from Hugging Face Cache...")
    cache_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        )
    )

    state_dict = {}
    safetensors_files = sorted(list(cache_dir.glob("*.safetensors")))
    pt_files = sorted(list(cache_dir.glob("*.pt"))) + sorted(list(cache_dir.glob("*.bin")))

    # 1. Load weights (Safetensors prioritized)
    if safetensors_files:
        for file in safetensors_files:
            state_dict.update(load_file(str(file), device="cpu"))
    elif pt_files:
        for file in pt_files:
            loaded = torch.load(file, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict) and "model" in loaded:
                loaded = loaded["model"]
            state_dict.update(loaded)
    else:
        raise FileNotFoundError(f"No valid weights (.safetensors, .pt, .bin) found in {cache_dir}")

    # 2. Convert dtype if specified
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

    # 3. Load into model structure
    print(f"Loading state_dict into model (strict={strict})...")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.to(device=device, dtype=dtype)
    model.eval()
    print("✓ Model successfully initialized and weights loaded.")
    return cache_dir