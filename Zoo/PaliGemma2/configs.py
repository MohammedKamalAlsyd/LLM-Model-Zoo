"""Consolidated configuration dataclasses for PaliGemma 2."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SigLipVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    projection_dim: int = 2304
    num_channels: int = 3
    image_size: int = 224
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0


@dataclass
class Gemma2Config:
    vocab_size: int = 257216
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    num_hidden_layers: int = 26
    sliding_window: int = 4096
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    query_pre_attn_scalar: float = 2304**-0.5
    attn_logit_softcapping: Optional[float] = 50.0
    final_logit_softcapping: Optional[float] = 30.0
    rope_theta: float = 10000.0
    head_dim: int = 256
    attention_bias: bool = False


@dataclass
class PaliGemma2Config:
    model_type: str = "paligemma"
    image_token_index: int = 257152
    projection_dim: int = 2304
    vision_config: SigLipVisionConfig = field(default_factory=SigLipVisionConfig)
    text_config: Gemma2Config = field(default_factory=Gemma2Config)

    @property
    def num_image_tokens(self) -> int:
        return (self.vision_config.image_size // self.vision_config.patch_size) ** 2


@dataclass
class PaliGemma2ProcessorConfig:
    image_size: int = 224
    image_seq_length: int = 256
    image_token: str = "<image>"
    num_location_tokens: int = 1024
    num_segmentation_tokens: int = 128
    
    # Preprocessing parameters from preprocessor_config.json
    rescale_factor: float = 1.0 / 255.0
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)