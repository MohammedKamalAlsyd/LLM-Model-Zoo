"""Consolidated configuration dataclasses for FLUX.1 [schnell] components."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import json
from pathlib import Path


@dataclass
class FluxVAEConfig:
    """Configuration for FLUX 16-channel AutoencoderKL."""
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    sample_size: int = 1024

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FluxVAEConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        if "block_out_channels" in filtered and isinstance(filtered["block_out_channels"], list):
            filtered["block_out_channels"] = tuple(filtered["block_out_channels"])
        return cls(**filtered)


@dataclass
class FluxSchedulerConfig:
    """Configuration for Flow-Matching Euler Discrete Scheduler."""
    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = True
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    time_shift_type: str = "exponential"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FluxSchedulerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class FluxTransformerConfig:
    """Configuration for the MMDiT 12B Dual/Single stream Transformer backbone."""
    patch_size: int = 1
    in_channels: int = 64  # 16 latent channels * 2x2 patch aggregation
    out_channels: Optional[int] = 64
    num_layers: int = 19  # Dual-stream MMDiT blocks
    num_single_layers: int = 38  # Single-stream DiT blocks
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096  # T5-XXL projection dimension
    pooled_projection_dim: int = 768  # CLIP-L pooled projection dimension
    guidance_embeds: bool = False  # False for Schnell, True for Dev
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)  # Sum = 128 (head_dim)
    theta: int = 10000

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FluxTransformerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        if "axes_dims_rope" in filtered and isinstance(filtered["axes_dims_rope"], list):
            filtered["axes_dims_rope"] = tuple(filtered["axes_dims_rope"])
        return cls(**filtered)


@dataclass
class T5Config:
    """Configuration for Google's T5-v1.1-XXL text encoder."""
    vocab_size: int = 32128
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "T5Config":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class FluxConfig:
    """Root configuration integrating AutoEncoder, Transformer, T5, CLIP, and Scheduler."""
    vae_config: FluxVAEConfig = field(default_factory=FluxVAEConfig)
    transformer_config: FluxTransformerConfig = field(default_factory=FluxTransformerConfig)
    scheduler_config: FluxSchedulerConfig = field(default_factory=FluxSchedulerConfig)
    t5_config: T5Config = field(default_factory=T5Config)
    clip_model_name: str = "openai/clip-vit-large-patch14"
    repo_id: str = "black-forest-labs/FLUX.1-schnell"
    default_image_size: int = 1024
    default_num_inference_steps: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FluxConfig":
        vae_dict = config_dict.get("vae_config", {})
        transformer_dict = config_dict.get("transformer_config", {})
        scheduler_dict = config_dict.get("scheduler_config", {})
        t5_dict = config_dict.get("t5_config", {})

        return cls(
            vae_config=FluxVAEConfig.from_dict(vae_dict) if isinstance(vae_dict, dict) else vae_dict,
            transformer_config=(
                FluxTransformerConfig.from_dict(transformer_dict)
                if isinstance(transformer_dict, dict)
                else transformer_dict
            ),
            scheduler_config=(
                FluxSchedulerConfig.from_dict(scheduler_dict)
                if isinstance(scheduler_dict, dict)
                else scheduler_dict
            ),
            t5_config=T5Config.from_dict(t5_dict) if isinstance(t5_dict, dict) else t5_dict,
            clip_model_name=config_dict.get("clip_model_name", "openai/clip-vit-large-patch14"),
            repo_id=config_dict.get("repo_id", "black-forest-labs/FLUX.1-schnell"),
            default_image_size=config_dict.get("default_image_size", 1024),
            default_num_inference_steps=config_dict.get("default_num_inference_steps", 4),
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "FluxConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)