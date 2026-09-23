"""Consolidated configuration dataclasses for CLIP models and processors."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path


@dataclass
class CLIPVisionConfig:
    """Configuration for the CLIP Vision Transformer backbone."""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    image_size: int = 224
    patch_size: int = 32
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    projection_dim: int = 512

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CLIPVisionConfig":
        """Instantiates a CLIPVisionConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


@dataclass
class CLIPTextConfig:
    """Configuration for the CLIP Text Transformer backbone."""
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_hidden_layers: int = 12
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5
    projection_dim: int = 512
    pad_token_id: int = 1
    eos_token_id: int = 49407
    bos_token_id: int = 49406

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CLIPTextConfig":
        """Instantiates a CLIPTextConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


@dataclass
class CLIPConfig:
    """Root configuration for dual-encoder CLIP model."""
    text_config: CLIPTextConfig = field(default_factory=CLIPTextConfig)
    vision_config: CLIPVisionConfig = field(default_factory=CLIPVisionConfig)
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CLIPConfig":
        """Constructs a CLIPConfig, parsing nested text and vision configurations."""
        vision_dict = config_dict.get("vision_config", {})
        text_dict = config_dict.get("text_config", {})

        vision_cfg = CLIPVisionConfig.from_dict(vision_dict) if isinstance(vision_dict, dict) else vision_dict
        text_cfg = CLIPTextConfig.from_dict(text_dict) if isinstance(text_dict, dict) else text_dict

        return cls(
            text_config=text_cfg,
            vision_config=vision_cfg,
            projection_dim=config_dict.get("projection_dim", 512),
            logit_scale_init_value=config_dict.get("logit_scale_init_value", 2.6592),
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "CLIPConfig":
        """Loads configuration directly from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class CLIPProcessorConfig:
    """Configuration for CLIP preprocessing pipelines."""
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    rescale_factor: float = 1.0 / 255.0
    max_text_length: int = 77
    do_resize: bool = True
    do_center_crop: bool = True
    do_rescale: bool = True
    do_normalize: bool = True
    do_convert_rgb: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CLIPProcessorConfig":
        """Instantiates processor configuration from a dictionary."""
        cfg_dict = dict(config_dict)
        if "image_mean" in cfg_dict and isinstance(cfg_dict["image_mean"], list):
            cfg_dict["image_mean"] = tuple(cfg_dict["image_mean"])
        if "image_std" in cfg_dict and isinstance(cfg_dict["image_std"], list):
            cfg_dict["image_std"] = tuple(cfg_dict["image_std"])

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys and v is not None}
        return cls(**filtered)