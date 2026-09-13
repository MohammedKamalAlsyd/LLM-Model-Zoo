# Copyright 2024-2025 The Alibaba Wan Team Authors and Project Contributors.
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from SubModels.CLIP import CLIPModel
from SubModels.DDPM import FlowDPMSolverMultistepScheduler, FlowUniPCMultistepScheduler, WanFlowScheduler
from SubModels.T5 import T5EncoderModel
from SubModels.UNet import VaceWanModel, WanModel
from SubModels.VAE import WanVAE

logger = logging.getLogger("WanModelLoader")

# Mapping of task and model scale to official Hugging Face repositories
HF_REPO_MAP = {
    "t2v-1.3B": "Wan-AI/Wan2.1-T2V-1.3B",
    "t2v-14B": "Wan-AI/Wan2.1-T2V-14B",
    "i2v-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P",
    "i2v-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P",
    "flf2v-14B": "Wan-AI/Wan2.1-I2V-14B-720P",
    "vace-1.3B": "Wan-AI/Wan2.1-T2V-1.3B",
    "vace-14B": "Wan-AI/Wan2.1-T2V-14B",
}

CONFIG_PRESETS = {
    "1.3B": {
        "dim": 1536,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "num_heads": 12,
        "num_layers": 30,
        "patch_size": (1, 2, 2),
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    },
    "14B": {
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "num_heads": 40,
        "num_layers": 40,
        "patch_size": (1, 2, 2),
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    },
}


@dataclass
class WanModelContainer:
    task: str
    scale: str
    vae: WanVAE
    text_encoder: T5EncoderModel
    clip: Optional[CLIPModel]
    dit: Union[WanModel, VaceWanModel]
    scheduler: WanFlowScheduler
    device: torch.device
    param_dtype: torch.dtype
    offload_model: bool
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    patch_size: Tuple[int, int, int] = (1, 2, 2)


def resolve_checkpoint_dir(checkpoint_path_or_repo: str) -> str:
    """
    Returns a valid local directory. If a HuggingFace repo ID is passed,
    it downloads or references the cached repository.
    """
    if os.path.isdir(checkpoint_path_or_repo):
        return checkpoint_path_or_repo

    logger.info(f"Downloading/verifying checkpoint from Hugging Face Hub: {checkpoint_path_or_repo}")
    local_dir = snapshot_download(
        repo_id=checkpoint_path_or_repo,
        allow_patterns=[
            "*.pth",
            "*.safetensors",
            "*.json",
            "google/umt5-xxl/*",
            "xlm-roberta-large/*",
        ],
    )
    return local_dir


def load_wan_submodels(
    task: str = "t2v",
    scale: str = "14B",
    checkpoint_dir: Optional[str] = None,
    resolution: str = "720P",
    device: Union[str, torch.device] = "cuda",
    param_dtype: torch.dtype = torch.bfloat16,
    offload_model: bool = True,
    t5_cpu: bool = False,
) -> WanModelContainer:
    """
    Unified entry point for loading all submodels for any Wan2.1 task.
    """
    device = torch.device(device)

    # 1. Resolve Checkpoint Directory
    if checkpoint_dir is None:
        key = f"{task}-{scale}"
        if task == "i2v":
            key = f"{task}-{scale}-{resolution}"
        repo_id = HF_REPO_MAP.get(key, HF_REPO_MAP["t2v-14B"])
        checkpoint_dir = resolve_checkpoint_dir(repo_id)
    else:
        checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)

    logger.info(f"Initializing Wan2.1 Pipeline [{task.upper()} - {scale}] from: {checkpoint_dir}")

    # 2. Instantiate VAE
    vae_path = os.path.join(checkpoint_dir, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        vae_path = hf_hub_download(repo_id=HF_REPO_MAP.get(f"t2v-{scale}", "Wan-AI/Wan2.1-T2V-14B"), filename="Wan2.1_VAE.pth")
    vae = WanVAE(z_dim=16, vae_pth=vae_path, device=device if not offload_model else "cpu", dtype=torch.float32)

    # 3. Instantiate UMT5-XXL Text Encoder
    t5_path = os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    tok_path = os.path.join(checkpoint_dir, "google/umt5-xxl")
    if not os.path.exists(t5_path):
        t5_path = hf_hub_download(repo_id=HF_REPO_MAP.get(f"t2v-{scale}", "Wan-AI/Wan2.1-T2V-14B"), filename="models_t5_umt5-xxl-enc-bf16.pth")
    if not os.path.exists(tok_path):
        tok_path = "google/umt5-xxl"

    t5_device = "cpu" if (t5_cpu or offload_model) else device
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=t5_device,
        checkpoint_path=t5_path,
        tokenizer_path=tok_path,
    )

    # 4. Instantiate CLIP Model (Only for I2V and FLF2V)
    clip = None
    if task in ("i2v", "flf2v"):
        clip_path = os.path.join(checkpoint_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        if not os.path.exists(clip_path):
            clip_path = hf_hub_download(repo_id=HF_REPO_MAP["i2v-14B-720P"], filename="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        clip_device = "cpu" if offload_model else device
        clip = CLIPModel(dtype=torch.float16, device=clip_device, checkpoint_path=clip_path)

    # 5. Instantiate Denoising Transformer (UNet / DiT)
    dit_cfg = dict(CONFIG_PRESETS[scale])
    model_type = "flf2v" if task == "flf2v" else ("i2v" if task == "i2v" else ("vace" if task == "vace" else "t2v"))
    
    # In dim calculation: T2V=16, I2V/FLF2V=36 (16 noise + 4 mask + 16 VAE)
    in_dim = 36 if task in ("i2v", "flf2v") else 16

    if task == "vace":
        dit = VaceWanModel(
            model_type="vace",
            in_dim=16,
            out_dim=16,
            text_dim=4096,
            **dit_cfg,
        )
    else:
        dit = WanModel(
            model_type=model_type,
            in_dim=in_dim,
            out_dim=16,
            text_dim=4096,
            **dit_cfg,
        )

    # Load DiT Weights (Safetensors or Binaries)
    dit_loaded = False
    for filename in ["diffusion_pytorch_model.safetensors", "Wan2.1_T2V_14B.pth", "Wan2.1_I2V_14B_720P.pth"]:
        full_path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(full_path):
            logger.info(f"Loading DiT backbone from {full_path}")
            if full_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(full_path)
            else:
                state_dict = torch.load(full_path, map_location="cpu")
            dit.load_state_dict(state_dict, strict=False)
            dit_loaded = True
            break

    if not dit_loaded:
        # Attempt loading via diffusers/transformers standard directory
        try:
            dit = WanModel.from_pretrained(checkpoint_dir)
        except Exception:
            logger.warning("Could not auto-load pre-packaged DiT checkpoint; initialized architecture with config.")

    dit = dit.to(dtype=param_dtype)
    if not offload_model:
        dit = dit.to(device)
    dit.eval().requires_grad_(False)

    # 6. Instantiate Flow-Matching Scheduler (DDPM equivalent)
    default_shift = 5.0 if resolution == "720P" else 3.0
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=default_shift)

    return WanModelContainer(
        task=task,
        scale=scale,
        vae=vae,
        text_encoder=text_encoder,
        clip=clip,
        dit=dit,
        scheduler=scheduler,
        device=device,
        param_dtype=param_dtype,
        offload_model=offload_model,
    )