"""Universal neural network building blocks, MLPs, and attention layers for SAM3."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def concat_padded_sequences(
    seq1: torch.Tensor,
    mask1: torch.Tensor,
    seq2: torch.Tensor,
    mask2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenates two variable-length padded token sequences and masks along sequence dimension.

    Args:
        seq1: Tensor of shape (batch_size, len1, dim).
        mask1: Boolean mask of shape (batch_size, len1) where True indicates valid tokens.
        seq2: Tensor of shape (batch_size, len2, dim).
        mask2: Boolean mask of shape (batch_size, len2).

    Returns:
        Concatenated sequence and combined valid mask.
    """
    bs, len1, d = seq1.shape
    _, len2, _ = seq2.shape
    actual_len1 = mask1.sum(dim=-1)
    actual_len2 = mask2.sum(dim=-1)
    final_lens = actual_len1 + actual_len2
    total_len = len1 + len2

    cat_mask = torch.arange(total_len, device=seq2.device)[None].repeat(bs, 1) < final_lens[:, None]
    cat_seq = torch.zeros((bs, total_len, d), device=seq2.device, dtype=seq2.dtype)
    cat_seq[:, :len1, :] = seq1

    scatter_idx = torch.arange(len2, device=seq2.device)[None].repeat(bs, 1) + actual_len1[:, None]
    cat_seq = cat_seq.scatter(1, scatter_idx[:, :, None].expand(-1, -1, d), seq2)
    return cat_seq, cat_mask


class Sam3MLP(nn.Module):
    """Two-layer Feed-Forward Network with optional dropout and configurable activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "gelu", dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = nn.GELU() if act == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.dropout(self.fc1(x))))


class Sam3DecoderMLP(nn.Module):
    """Multi-layer projection network matching SAM3 detector head checkpoints."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        if num_layers == 2:
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3: Optional[nn.Linear] = None
        else:
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        if self.layer3 is not None:
            x = F.relu(self.layer2(x))
            return self.layer3(x)
        return self.layer2(x)


class Sam3Attention(nn.Module):
    """Standard Multi-Head Attention leveraging FlashAttention / SDPA."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        b, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(b, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, k_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, k_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None and attention_mask.dtype != q.dtype:
            attention_mask = attention_mask.to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.hidden_size)
        return self.o_proj(out), None


def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    """Interleaved pairwise coordinate rotation for 2D RoPE."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb_2d(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotates spatial feature queries and keys according to 2D frequency bands."""
    q_out = (q.float() * cos) + (rotate_pairwise(q.float()) * sin)
    k_out = (k.float() * cos) + (rotate_pairwise(k.float()) * sin)
    return q_out.type_as(q), k_out.type_as(k)


class Sam3ViTRotaryEmbedding(nn.Module):
    """Computes separate height and width RoPE frequencies for ViT windowed grids."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int = 64, rope_theta: float = 10000.0) -> None:
        super().__init__()
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_expanded = position_ids[..., None].float()
        freqs = pos_expanded * self.inv_freq.float()
        cos, sin = freqs.cos(), freqs.sin()

        freq_w_cos, freq_h_cos = cos[:, 0], cos[:, 1]
        cos_hw = torch.cat([freq_w_cos, freq_h_cos], dim=-1)[None, ...].repeat_interleave(2, dim=-1)

        freq_w_sin, freq_h_sin = sin[:, 0], sin[:, 1]
        sin_hw = torch.cat([freq_w_sin, freq_h_sin], dim=-1)[None, ...].repeat_interleave(2, dim=-1)
        return cos_hw, sin_hw