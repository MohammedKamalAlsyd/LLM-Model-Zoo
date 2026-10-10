"""Universal Transformer encoder blocks supporting arbitrary activations and custom masks."""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from Zoo.Common.activations import get_activation


class CLIPAttention(nn.Module):
    """Multi-Head Self-Attention module supporting boolean causal or 4D custom masks."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            causal_attention_mask: Whether to apply causal autoregressive masking.

        Returns:
            Projected attention tensor of shape (batch_size, seq_len, hidden_size).
        """
        b, s, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch native scaled dot-product attention
        if attention_mask is not None:
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=causal_attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, s, self.embed_dim)
        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    """Two-layer feedforward network with configurable activation function."""

    def __init__(self, hidden_size: int, intermediate_size: int, act: Union[str, nn.Module] = "quick_gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = get_activation(act)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    """Transformer encoder block with pre-LayerNorm residual architecture."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layer_norm_eps: float = 1e-5,
        act: Union[str, nn.Module] = "quick_gelu",
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, act=act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention Residual
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class CLIPEncoder(nn.Module):
    """Transformer encoder stack mapping to checkpoint key `encoder.layers.*`."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        layer_norm_eps: float = 1e-5,
        act: Union[str, nn.Module] = "quick_gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(hidden_size, intermediate_size, num_heads, layer_norm_eps, act=act)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                causal_attention_mask=causal_attention_mask,
                attention_mask=attention_mask,
            )
        return hidden_states