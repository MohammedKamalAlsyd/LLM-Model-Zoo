import torch
from huggingface_hub import hf_hub_download
from Zoo.CLIP.CLIP import CLIPModel

def load_clip_from_hf(repo_id: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> CLIPModel:
    """
    Downloads and loads the pretrained weights from Hugging Face into our custom architecture.
    """
    print(f"Initializing Custom CLIP Architecture on {device.upper()}...")
    model = CLIPModel()
    
    print(f"Downloading weights from HF: {repo_id}...")
    weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    
    print("Loading weights into model (strict=True)...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    # Because we perfectly mapped the variable names (e.g. vision_model.encoder.layers),
    # this will load flawlessly without needing a conversion script!
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    print("Model successfully loaded!")
    
    return model