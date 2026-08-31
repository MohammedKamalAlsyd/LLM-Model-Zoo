import torch
from torch import nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0 native Flash Attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=causal_attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = QuickGELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))

class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: bool = False) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class CLIPEncoder(nn.Module):
    """Wrapper class to hold the ModuleList and satisfy both Pylance and HF weights."""
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, intermediate_size, num_heads) 
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, causal_attention_mask: bool = False) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_attention_mask)
        return hidden_states