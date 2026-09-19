"""Universal Rotary Position Embedding (RoPE) functions."""

from typing import Tuple
import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE rotation to query and key tensors.

    Args:
        q: Query tensor of shape (B, H, S, D) or (B, S, H, D).
        k: Key tensor of shape (B, H_kv, S, D) or (B, S, H_kv, D).
        cos, sin: Frequency embeddings of shape (B, S, D).
        unsqueeze_dim: Dimension along which to broadcast heads (default 1 for B, H, S, D).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed