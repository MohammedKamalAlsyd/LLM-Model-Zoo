"""Configuration dataclasses for Ministral-3 Multimodal models.

Defines decoupled configurations for the Pixtral vision encoder,
the Ministral-3 language backbone, and the unified multimodal wrapper.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class PixtralVisionConfig:
    """Configuration for the Pixtral Vision Encoder."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 1540
    patch_size: int = 14
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    head_dim: int = 64

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PixtralVisionConfig":
        """Instantiates a PixtralVisionConfig from a dictionary, extracting nested RoPE parameters."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Flatten rope_parameters if present
        if "rope_parameters" in config_dict and isinstance(config_dict["rope_parameters"], dict):
            if "rope_theta" in config_dict["rope_parameters"]:
                filtered["rope_theta"] = float(config_dict["rope_parameters"]["rope_theta"])

        return cls(**filtered)


@dataclass
class Ministral3TextConfig:
    """Configuration for the Ministral-3 Autoregressive Language Model."""

    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-05
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    sliding_window: Optional[int] = None
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
            "rope_theta": 1000000.0,
        }
    )
    pad_token_id: int = 11
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Ministral3TextConfig":
        """Instantiates a Ministral3TextConfig from a dictionary with key filtering."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Ministral3MultimodalConfig:
    """Master configuration binding Pixtral Vision, Projector, and Ministral-3 Text."""

    vision_config: PixtralVisionConfig = field(default_factory=PixtralVisionConfig)
    text_config: Ministral3TextConfig = field(default_factory=Ministral3TextConfig)
    image_token_index: int = 10
    spatial_merge_size: int = 2
    vision_feature_layer: int = -1
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Ministral3MultimodalConfig":
        """Instantiates master multimodal config from official config.json dictionaries."""
        vision_dict = config_dict.get("vision_config", {})
        text_dict = config_dict.get("text_config", {})

        vision_cfg = PixtralVisionConfig.from_dict(vision_dict)
        text_cfg = Ministral3TextConfig.from_dict(text_dict)

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {
            k: v for k, v in config_dict.items()
            if k in valid_keys and k not in ("vision_config", "text_config")
        }

        return cls(
            vision_config=vision_cfg,
            text_config=text_cfg,
            **filtered,
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "Ministral3MultimodalConfig":
        """Loads and parses an official configuration JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)