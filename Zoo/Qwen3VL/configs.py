"""Configuration dataclasses for Qwen3-VL models.

Defines decoupled configurations for the Vision Encoder (with DeepStack indices),
the 3D M-RoPE Language Backbone, and the Master Multimodal Wrapper.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class Qwen3VLVisionConfig:
    """Configuration for the Qwen3-VL Dynamic Resolution Vision Encoder."""

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    out_hidden_size: int = 2560
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [5, 11, 17])

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3VLVisionConfig":
        """Constructs Qwen3VLVisionConfig filtering out extra keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Qwen3VLTextConfig:
    """Configuration for the Qwen3-VL Autoregressive Language Model."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5000000.0
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3VLTextConfig":
        """Constructs Qwen3VLTextConfig filtering out extra keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Qwen3VLConfig:
    """Master configuration binding Vision, DeepStack, and Language Backbones."""

    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3VLConfig":
        """Instantiates master configuration from nested dictionary."""
        vision_dict = config_dict.get("vision_config", {})
        text_dict = config_dict.get("text_config", {})

        vision_cfg = (
            Qwen3VLVisionConfig.from_dict(vision_dict)
            if isinstance(vision_dict, dict)
            else vision_dict
        )
        text_cfg = (
            Qwen3VLTextConfig.from_dict(text_dict)
            if isinstance(text_dict, dict)
            else text_dict
        )

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {
            k: v
            for k, v in config_dict.items()
            if k in valid_keys and k not in ("vision_config", "text_config")
        }

        return cls(
            vision_config=vision_cfg,
            text_config=text_cfg,
            **filtered,
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "Qwen3VLConfig":
        """Loads configuration from an official JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)