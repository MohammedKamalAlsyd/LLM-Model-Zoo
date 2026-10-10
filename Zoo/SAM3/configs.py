"""Consolidated configuration dataclasses for SAM3 components and preprocessing."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union
import json
from pathlib import Path
from Zoo.CLIP.configs import CLIPTextConfig



@dataclass
class SAM3VisionConfig:
    """Configuration for the SAM3 ViT Backbone and FPN Neck."""
    patch_size: int = 14
    in_channels: int = 3
    hidden_size: int = 1024
    intermediate_size: int = 4736
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    pretrain_patches: int = 576
    window_size: int = 24
    global_attn_indexes: Tuple[int, ...] = (7, 15, 23, 31)
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    fpn_dim: int = 256
    fpn_scales: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAM3VisionConfig":
        """Instantiates a SAM3VisionConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        for k in ("global_attn_indexes", "fpn_scales"):
            if k in filtered and isinstance(filtered[k], list):
                filtered[k] = tuple(filtered[k])
        return cls(**filtered)


@dataclass
class SAM3TextConfig(CLIPTextConfig):
    """Configuration for the SAM3 CLIP-based text prompt backbone."""
    vocab_size: int = 49408
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    max_position_embeddings: int = 32
    projection_dim: int = 512
    detr_proj_dim: int = 256
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAM3TextConfig":
        """Instantiates a SAM3TextConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SAM3DETRConfig:
    """Configuration for the Multimodal DETR Encoder-Decoder and scoring."""
    hidden_size: int = 256
    intermediate_size: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    num_queries: int = 200
    dropout: float = 0.1
    roi_size: int = 7

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAM3DETRConfig":
        """Instantiates a SAM3DETRConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SAM3TrackerConfig:
    """Configuration for the SAM3 Video Memory Tracking engine."""
    hidden_size: int = 256
    inner_dim: int = 128
    mem_dim: int = 64
    num_mask_tokens: int = 4
    num_layers: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAM3TrackerConfig":
        """Instantiates a SAM3TrackerConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SAM3ProcessorConfig:
    """Configuration for image, video, and prompt preprocessing pipelines."""
    target_size: int = 1008
    max_text_length: int = 32
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    default_confidence_threshold: float = 0.35
    mask_threshold: float = 0.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAM3ProcessorConfig":
        """Instantiates a SAM3ProcessorConfig from a dictionary."""
        cfg = dict(config_dict)
        if "image_mean" in cfg and isinstance(cfg["image_mean"], list):
            cfg["image_mean"] = tuple(cfg["image_mean"])
        if "image_std" in cfg and isinstance(cfg["image_std"], list):
            cfg["image_std"] = tuple(cfg["image_std"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys and v is not None}
        return cls(**filtered)


@dataclass
class SAM3Config:
    """Root configuration integrating Vision, Text, DETR, and Tracker modules."""
    vision_config: SAM3VisionConfig = field(default_factory=SAM3VisionConfig)
    text_config: SAM3TextConfig = field(default_factory=SAM3TextConfig)
    detr_config: SAM3DETRConfig = field(default_factory=SAM3DETRConfig)
    tracker_config: SAM3TrackerConfig = field(default_factory=SAM3TrackerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAM3Config":
        """Constructs a SAM3Config, recursively parsing nested configurations."""
        v_dict = config_dict.get("vision_config", {})
        t_dict = config_dict.get("text_config", {})
        d_dict = config_dict.get("detr_config", {})
        tr_dict = config_dict.get("tracker_config", {})

        return cls(
            vision_config=SAM3VisionConfig.from_dict(v_dict) if isinstance(v_dict, dict) else v_dict,
            text_config=SAM3TextConfig.from_dict(t_dict) if isinstance(t_dict, dict) else t_dict,
            detr_config=SAM3DETRConfig.from_dict(d_dict) if isinstance(d_dict, dict) else d_dict,
            tracker_config=SAM3TrackerConfig.from_dict(tr_dict) if isinstance(tr_dict, dict) else tr_dict,
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "SAM3Config":
        """Loads configuration directly from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)