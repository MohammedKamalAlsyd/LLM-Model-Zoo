import torch
import torch.nn as nn
from dataclasses import dataclass
from Pixtral import PixtralConfig

@dataclass
class Ministral3Config:
    attention_dropout: float = 0.0
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 14336
    max_position_embeddings: int = 262144
    num_attention_heads: int = 32
    num_hidden_layers: int = 34
    num_key_value_heads: int = 8 # For GQA
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

@dataclass
class Mistral3Config:
    spatial_merge_size: int = 2
    vocab_size: int = 131072
    text_config: Ministral3Config = Ministral3Config()
    vision_config: PixtralConfig = PixtralConfig()



    
    
