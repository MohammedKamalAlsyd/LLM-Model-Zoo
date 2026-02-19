import torch
import torch.nn as nn
from dataclasses import dataclass
from Pixtral import PixtralConfig

@dataclass
class Mistral3TextConfig:
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
    text_config: Mistral3TextConfig = Mistral3TextConfig()
    vision_config: PixtralConfig = PixtralConfig()


class Mistral3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        org_input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(org_input_dtype)
    
    
