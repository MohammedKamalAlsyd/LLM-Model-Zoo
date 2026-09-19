"""Consolidated configuration dataclasses for PaliGemma 2 models and processors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path


@dataclass
class SigLipVisionConfig:
    """Configuration for the SigLIP vision transformer tower."""
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    projection_dim: int = 3584
    num_channels: int = 3
    image_size: int = 448
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_positions: int = 1024
    num_image_tokens: int = 1024
    vision_use_head: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SigLipVisionConfig":
        """Instantiates a SigLipVisionConfig from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


@dataclass
class Gemma2Config:
    """Configuration for the Gemma 2 autoregressive language model."""
    vocab_size: int = 257216
    hidden_size: int = 3584
    intermediate_size: int = 14336
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_hidden_layers: int = 42
    sliding_window: int = 4096
    pad_token_id: int = 0
    eos_token_id: Union[int, List[int], Tuple[int, ...]] = (1, 107)
    bos_token_id: int = 2
    query_pre_attn_scalar: float = 256.0
    attn_logit_softcapping: Optional[float] = 50.0
    final_logit_softcapping: Optional[float] = 30.0
    rope_theta: float = 10000.0
    head_dim: int = 256
    attention_bias: bool = False
    hidden_activation: str = "gelu_pytorch_tanh"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Gemma2Config":
        """Instantiates a Gemma2Config from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        if "eos_token_id" in filtered_dict and isinstance(filtered_dict["eos_token_id"], list):
            filtered_dict["eos_token_id"] = tuple(filtered_dict["eos_token_id"])
        return cls(**filtered_dict)


@dataclass
class PaliGemma2Config:
    """Root configuration for PaliGemma 2 conditional generation."""
    model_type: str = "paligemma"
    image_token_index: int = 257152
    projection_dim: int = 3584
    vision_config: SigLipVisionConfig = field(default_factory=SigLipVisionConfig)
    text_config: Gemma2Config = field(default_factory=Gemma2Config)

    @property
    def num_image_tokens(self) -> int:
        return (self.vision_config.image_size // self.vision_config.patch_size) ** 2

    @property
    def eos_token_ids(self) -> Tuple[int, ...]:
        """Returns eos_token_ids as a tuple of integers."""
        eos = self.text_config.eos_token_id
        if isinstance(eos, (int, float)):
            return (int(eos),)
        return tuple(int(x) for x in eos)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PaliGemma2Config":
        """Constructs a PaliGemma2Config, parsing nested text and vision configurations."""
        vision_dict = config_dict.get("vision_config", {})
        text_dict = config_dict.get("text_config", {})

        vision_cfg = SigLipVisionConfig.from_dict(vision_dict) if isinstance(vision_dict, dict) else vision_dict
        text_cfg = Gemma2Config.from_dict(text_dict) if isinstance(text_dict, dict) else text_dict

        return cls(
            model_type=config_dict.get("model_type", "paligemma"),
            image_token_index=config_dict.get("image_token_index", 257152),
            projection_dim=config_dict.get("projection_dim", 3584),
            vision_config=vision_cfg,
            text_config=text_cfg,
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "PaliGemma2Config":
        """Loads configuration directly from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PaliGemma2ProcessorConfig:
    """Configuration for image preprocessing and tokenizer special tokens."""
    image_size: int = 448
    image_seq_length: int = 1024
    image_token: str = "<image>"
    num_location_tokens: int = 1024
    num_segmentation_tokens: int = 128

    # Pipeline toggle flags matching HF preprocessor_config.json
    do_resize: bool = True
    do_rescale: bool = True
    do_normalize: bool = True
    do_convert_rgb: bool = True
    resample: int = 3  # Image.Resampling.BICUBIC
    rescale_factor: float = 1.0 / 255.0
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PaliGemma2ProcessorConfig":
        """Constructs a PaliGemma2ProcessorConfig adapting HF dictionary formats."""
        cfg_dict = dict(config_dict)

        if "size" in cfg_dict and isinstance(cfg_dict["size"], dict):
            cfg_dict["image_size"] = cfg_dict["size"].get("height", 448)

        if "image_mean" in cfg_dict and isinstance(cfg_dict["image_mean"], list):
            cfg_dict["image_mean"] = tuple(cfg_dict["image_mean"])

        if "image_std" in cfg_dict and isinstance(cfg_dict["image_std"], list):
            cfg_dict["image_std"] = tuple(cfg_dict["image_std"])

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys and v is not None}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "PaliGemma2ProcessorConfig":
        """Loads processor configuration from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)