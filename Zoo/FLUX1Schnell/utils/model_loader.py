import os
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_safetensors_shards(repo_id: str, subfolder: str, device: str = "cpu") -> dict:
    """
    Downloads and combines safetensors shards for a given subfolder from Hugging Face.
    """
    state_dict = {}

    # Check for single file or sharded files
    try:
        # Single file case (e.g. VAE)
        file_path = hf_hub_download(repo_id=repo_id, filename="diffusion_pytorch_model.safetensors", subfolder=subfolder)
        return load_file(file_path, device=device)
    except Exception:
        pass

    try:
        file_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", subfolder=subfolder)
        return load_file(file_path, device=device)
    except Exception:
        pass

    # Sharded files case (e.g. Transformer & T5)
    # Scan from shard 1 to 10
    found_any = False
    for i in range(1, 15):
        pattern_candidates = [
            f"diffusion_pytorch_model-{i:05d}-of-*.safetensors",
            f"model-{i:05d}-of-*.safetensors",
        ]
        shard_loaded = False
        for pattern in pattern_candidates:
            try:
                # Try common total shard counts: 2, 3, 4, 5
                for total in [2, 3, 4, 5]:
                    cand_name = pattern.replace("*", f"{total:05d}")
                    try:
                        file_path = hf_hub_download(repo_id=repo_id, filename=cand_name, subfolder=subfolder)
                        print(f"Loading shard: {cand_name}...")
                        shard_dict = load_file(file_path, device=device)
                        state_dict.update(shard_dict)
                        shard_loaded = True
                        found_any = True
                        break
                    except Exception:
                        continue
                if shard_loaded:
                    break
            except Exception:
                continue

    if not found_any:
        raise FileNotFoundError(f"Could not locate safetensors weights for {subfolder} in {repo_id}")

    return state_dict


def load_flux_transformer_weights(transformer: torch.nn.Module, repo_id: str, device: str = "cpu"):
    print("Downloading/Loading Flux Transformer weights (~24 GB)...")
    state_dict = load_safetensors_shards(repo_id=repo_id, subfolder="transformer", device=device)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Transformer weights loaded! (missing: {len(missing)}, unexpected: {len(unexpected)})")


def load_flux_vae_weights(vae: torch.nn.Module, repo_id: str, device: str = "cpu"):
    print("Downloading/Loading VAE weights (~335 MB)...")
    state_dict = load_safetensors_shards(repo_id=repo_id, subfolder="vae", device=device)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f"VAE weights loaded! (missing: {len(missing)}, unexpected: {len(unexpected)})")


def load_flux_t5_weights(t5: torch.nn.Module, repo_id: str, device: str = "cpu"):
    print("Downloading/Loading T5-XXL weights (~9.5 GB)...")
    state_dict = load_safetensors_shards(repo_id=repo_id, subfolder="text_encoder_2", device=device)
    t5.load_hf_weights(state_dict)
    print("T5-XXL weights loaded!")