"""Consolidated configuration dataclasses for Stable Diffusion v1.5 components."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from Zoo.CLIP.configs import CLIPTextConfig


@dataclass
class VAEConfig:
    """Configuration for the First-Stage Autoencoder (VAE)."""
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks_per_stage: int = 2
    scaling_factor: float = 0.18215
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VAEConfig":
        """Instantiates a VAEConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        if "channel_multipliers" in filtered and isinstance(filtered["channel_multipliers"], list):
            filtered["channel_multipliers"] = tuple(filtered["channel_multipliers"])
        return cls(**filtered)


@dataclass
class UNetConfig:
    """Configuration for the Noise Prediction Latent UNet."""
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 320
    time_embed_dim: int = 1280
    context_dim: int = 768  # Text projection dimension from CLIP
    num_heads: int = 8
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    attention_head_dim: Tuple[int, ...] = (40, 80, 160, 160)
    num_res_blocks: int = 2
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UNetConfig":
        """Instantiates a UNetConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        for k in ("channel_multipliers", "attention_head_dim"):
            if k in filtered and isinstance(filtered[k], list):
                filtered[k] = tuple(filtered[k])
        return cls(**filtered)


@dataclass
class DDPMConfig:
    """Configuration for the Denoising Diffusion Probabilistic Models (DDPM) Scheduler."""
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.0120
    beta_schedule: str = "scaled_linear"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DDPMConfig":
        """Instantiates a DDPMConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class StableDiffusionConfig:
    """Root configuration integrating VAE, UNet, CLIP Text Tower, and DDPM Scheduler."""
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    unet_config: UNetConfig = field(default_factory=UNetConfig)
    text_config: CLIPTextConfig = field(
        default_factory=lambda: CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_hidden_layers=12,
            max_position_embeddings=77,
            projection_dim=768,
        )
    )
    ddpm_config: DDPMConfig = field(default_factory=DDPMConfig)
    clip_model_name: str = "openai/clip-vit-large-patch14"
    image_size: int = 512

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StableDiffusionConfig":
        """Constructs a StableDiffusionConfig, recursively parsing sub-configs."""
        vae_dict = config_dict.get("vae_config", {})
        unet_dict = config_dict.get("unet_config", {})
        text_dict = config_dict.get("text_config", {})
        ddpm_dict = config_dict.get("ddpm_config", {})

        return cls(
            vae_config=VAEConfig.from_dict(vae_dict) if isinstance(vae_dict, dict) else vae_dict,
            unet_config=UNetConfig.from_dict(unet_dict) if isinstance(unet_dict, dict) else unet_dict,
            text_config=CLIPTextConfig.from_dict(text_dict) if isinstance(text_dict, dict) else text_dict,
            ddpm_config=DDPMConfig.from_dict(ddpm_dict) if isinstance(ddpm_dict, dict) else ddpm_dict,
            clip_model_name=config_dict.get("clip_model_name", "openai/clip-vit-large-patch14"),
            image_size=config_dict.get("image_size", 512),
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "StableDiffusionConfig":
        """Loads configuration from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)