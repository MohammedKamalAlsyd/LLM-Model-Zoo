import os
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from Zoo.SAM3.SAM3 import Sam3Model

def load_sam3_from_hf(
    repo_id: str = "facebook/sam3",
    checkpoint_file: str = "model.safetensors",
    device: str = "cpu"
) -> Sam3Model:
    """
    Downloads pretrained weights from Hugging Face and loads them directly
    into our custom architecture using strict=True.
    """
    print(f"Initializing SAM3 Architecture on {device.upper()}...")
    model = Sam3Model()

    print(f"Downloading checkpoint: {repo_id}/{checkpoint_file}...")
    file_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_file)

    print("Loading weights into model (strict=True)...")
    if file_path.endswith(".safetensors"):
        state_dict = load_file(file_path, device="cpu")
    else:
        state_dict = torch.load(file_path, map_location="cpu", weights_only=True)

    # Direct 1:1 load: perfectly identical key hierarchy
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    print("✓ SAM3 Model loaded successfully without key mapping!")
    return model