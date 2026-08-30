import torch
from Zoo.StableDiffusion.SubModels.CLIP import CLIP
from Zoo.StableDiffusion.SubModels.UNet import UNetModel
from Zoo.StableDiffusion.SubModels.VAE import VAE

def load_models_from_standard_weights(ckpt_path: str, device: str = "cpu") -> dict:
    print(f"Loading weights from {ckpt_path}...")
    
    # Load the full checkpoint into RAM (using CPU initially to save VRAM)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # SD 1.5 checkpoints nest the weights under a "state_dict" key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Initialize empty architectures
    print("Initializing architectures...")
    unet = UNetModel()
    vae = VAE()
    clip = CLIP()

    # Create empty dictionaries for our sub-models
    unet_dict = {}
    vae_dict = {}
    clip_dict = {}

    print("Routing weights to sub-models...")
    # Route weights based on their original prefixes
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model."):
            unet_dict[key.replace("model.diffusion_model.", "")] = value
            
        elif key.startswith("first_stage_model."):
            vae_dict[key.replace("first_stage_model.", "")] = value
            
        elif key.startswith("cond_stage_model.transformer.text_model."):
            clip_dict[key.replace("cond_stage_model.transformer.text_model.", "")] = value

    clip_dict.pop("embeddings.position_ids", None)
    print("Loading weights into modules...")
    # strict=True ensures we didn't miss any keys and no layers are left randomly initialized
    unet.load_state_dict(unet_dict, strict=True)
    vae.load_state_dict(vae_dict, strict=True)
    clip.load_state_dict(clip_dict, strict=True)

    print("Model successfully loaded into RAM!")
    
    # Return dictionary expected by serve.py
    return {
        "clip": clip,
        "vae": vae,    # VAE handles both encoding and decoding
        "unet": unet
    }