import os
from glob import glob
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from ..config import Qwen3VLConfig
from ..Qwen3VLMultimodal import Qwen3VLForConditionalGeneration


def load_qwen3_vl(model_path_or_repo: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Qwen3VLForConditionalGeneration:
    """Loads weights into standalone Qwen3VL model."""
    if not os.path.isdir(model_path_or_repo):
        model_dir = snapshot_download(repo_id=model_path_or_repo)
    else:
        model_dir = model_path_or_repo

    config = Qwen3VLConfig()
    model = Qwen3VLForConditionalGeneration(config).to(dtype=dtype)

    safetensor_files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device="cpu"))

    # Remap HuggingFace nested keys to our standalone module hierarchy
    remapped_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # Strip "model." prefix from HF hierarchy
        if new_k.startswith("model.visual."):
            new_k = new_k.replace("model.visual.", "visual.")
        elif new_k.startswith("model.language_model."):
            new_k = new_k.replace("model.language_model.", "language_model.")
        elif new_k.startswith("model."):
            new_k = new_k.replace("model.", "")

        remapped_state_dict[new_k] = v

    model.load_state_dict(remapped_state_dict, strict=False)
    model.to(device=device)
    model.eval()
    return model