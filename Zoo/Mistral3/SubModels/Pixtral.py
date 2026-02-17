from torch import nn
import torch
from dataclasses import dataclass


@dataclass
class PixtralConfig:
    head_dim: int = 64
    num_heads: int = 16
    attention_dropout: float = 0.0
    hidden_size: int = 1024
    image_size: int = 1540
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    rope_theta: float = 100000.0


class PixtralMLP(nn.Module):
    def __init__(self, config: PixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(
            self.act_fn(self.up_proj(hidden_states) * self.gate_proj(hidden_states))
        )
        return down_proj


class PixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(
            torch.float32
        )  # shape: [batch_size, seq_len, hidden_size]
        variance = hidden_states.pow(2).mean(
            -1, keepdim=True
        )  # shape: [batch_size, seq_len, 1]
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


