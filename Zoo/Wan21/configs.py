"""Configuration dataclasses for the Wan2.1 Video Generation System.

Defines decoupled configurations for the Wan 3D Causal VAE, UMT5-XXL text encoder,
XLM-RoBERTa CLIP visual encoder, WanModel 3D DiT, and Continuous Flow Schedulers.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class WanVAEConfig:
    """Configuration for the Wan 3D Causal Convolutional VAE."""

    dim: int = 96
    z_dim: int = 16
    dim_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_scales: List[float] = field(default_factory=list)
    temperal_downsample: List[bool] = field(default_factory=lambda: [False, True, True])
    dropout: float = 0.0
    cache_t: int = 2
    stride: Tuple[int, int, int] = (4, 8, 8)
    mean: List[float] = field(
        default_factory=lambda: [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]
    )
    std: List[float] = field(
        default_factory=lambda: [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
    )

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "WanVAEConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


@dataclass
class WanT5Config:
    """Configuration for the UMT5-XXL text encoder."""

    vocab_size: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    max_dist: int = 128
    shared_pos: bool = False
    dropout: float = 0.1
    text_len: int = 512
    tokenizer_path: str = "google/umt5-xxl"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "WanT5Config":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


@dataclass
class WanCLIPConfig:
    """Configuration for the XLM-RoBERTa ViT-H/14 visual encoder."""

    image_size: int = 224
    patch_size: int = 14
    dim: int = 1280
    mlp_ratio: float = 4.0
    num_heads: int = 16
    num_layers: int = 32
    norm_eps: float = 1e-5
    extract_layer_idx: int = 31

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "WanCLIPConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


@dataclass
class WanDiTConfig:
    """Configuration for the WanModel 3D Denoising Transformer (DiT)."""

    model_type: str = "t2v"
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 40
    num_layers: int = 40
    window_size: Tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_theta: float = 10000.0
    chunk_size: int = 1024

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "WanDiTConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


@dataclass
class WanSchedulerConfig:
    """Configuration for the FlowUniPC and FlowDPM++ continuous ODE schedulers."""

    num_train_timesteps: int = 1000
    solver_order: int = 2
    shift: float = 3.0
    prediction_type: str = "flow_prediction"
    use_dynamic_shifting: bool = False
    solver_type: str = "bh2"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "WanSchedulerConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in valid})


@dataclass
class Wan21Config:
    """Master configuration binding VAE, UMT5, CLIP, DiT, and Flow Schedulers."""

    task: str = "t2v"
    scale: str = "14B"
    vae_config: WanVAEConfig = field(default_factory=WanVAEConfig)
    text_config: WanT5Config = field(default_factory=WanT5Config)
    clip_config: WanCLIPConfig = field(default_factory=WanCLIPConfig)
    dit_config: WanDiTConfig = field(default_factory=WanDiTConfig)
    scheduler_config: WanSchedulerConfig = field(default_factory=WanSchedulerConfig)
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    default_resolution: Tuple[int, int] = (832, 480)
    fps: int = 16

    @classmethod
    def from_preset(cls, task: str = "t2v", scale: str = "1.3B", resolution: str = "480P") -> "Wan21Config":
        """Instantiates an official Wan2.1 configuration preset."""
        presets = {
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

        dit_params = dict(presets.get(scale, presets["1.3B"]))
        dit_params["model_type"] = task
        # T2V takes 16 latent noise channels; I2V and FLF2V take 36 (16 noise + 4 mask + 16 VAE)
        dit_params["in_dim"] = 36 if task in ("i2v", "flf2v") else 16

        shift_val = 5.0 if resolution == "720P" else 3.0

        return cls(
            task=task,
            scale=scale,
            dit_config=WanDiTConfig.from_dict(dit_params),
            scheduler_config=WanSchedulerConfig(shift=shift_val),
        )

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "Wan21Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            task=data.get("task", "t2v"),
            scale=data.get("scale", "1.3B"),
            vae_config=WanVAEConfig.from_dict(data.get("vae_config", {})),
            text_config=WanT5Config.from_dict(data.get("text_config", {})),
            clip_config=WanCLIPConfig.from_dict(data.get("clip_config", {})),
            dit_config=WanDiTConfig.from_dict(data.get("dit_config", {})),
            scheduler_config=WanSchedulerConfig.from_dict(data.get("scheduler_config", {})),
        )