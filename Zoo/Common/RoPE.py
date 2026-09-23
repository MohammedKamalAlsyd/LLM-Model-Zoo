"""Universal Rotary Position Embedding (RoPE) functions."""

import math
from typing import Any, Dict, Optional, Tuple
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

def compute_yarn_parameters(
    head_dim: int,
    base: float = 10000.0,
    factor: float = 16.0,
    original_max_position_embeddings: int = 16384,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Computes YaRN (Yet another RoPE extensioN) inverse frequencies and attention scaling factor.

    Splits the frequency spectrum into:
      - High frequencies: Extrapolated (unmodified standard RoPE)
      - Low frequencies: Interpolated (scaled by factor)
      - Mid frequencies: Smoothly blended via linear ramp

    Args:
        head_dim: Attention head dimension (e.g. 64, 128, 256).
        base: RoPE base theta (e.g. 10000.0, 1000000.0).
        factor: Context window extension multiplier (e.g. 16.0 for 16x context).
        original_max_position_embeddings: Pretrained context length before scaling.
        beta_fast: Upper bound for frequency extrapolation ramp.
        beta_slow: Lower bound for frequency interpolation ramp.
        mscale: Magnitude scaling coefficient.
        mscale_all_dim: Global magnitude scaling coefficient.
        device: Optional target device for the returned tensor.

    Returns:
        inv_freq: Tensor of shape (head_dim // 2,) containing blended inverse frequencies.
        attention_scaling: Float scalar multiplier applied to cos and sin embeddings.
    """
    pos_freqs = base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs

    if factor <= 1.0:
        return inv_freq_extrapolation, 1.0

    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    def find_dim(rotations: float) -> float:
        """Finds the frequency index corresponding to a given rotation threshold."""
        return (head_dim * math.log(original_max_position_embeddings / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low_bound = max(find_dim(beta_fast), 0.0)
    high_bound = min(find_dim(beta_slow), float(head_dim // 2 - 1))

    dim_indices = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
    if high_bound == low_bound:
        high_bound += 0.001  # Prevent division by zero

    ramp = torch.clamp((dim_indices - low_bound) / (high_bound - low_bound), 0.0, 1.0)

    # High freq -> extrapolate (ramp=0); Low freq -> interpolate (ramp=1)
    inv_freq = inv_freq_extrapolation * (1.0 - ramp) + inv_freq_interpolation * ramp

    def get_mscale(scale: float, coeff: float) -> float:
        return 0.1 * coeff * math.log(scale) + 1.0 if scale > 1.0 else 1.0

    attention_scaling = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
    return inv_freq, float(attention_scaling)


def compute_rope_parameters(
    head_dim: int,
    base: float = 10000.0,
    rope_type: str = "default",
    rope_parameters: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Universal dispatcher for RoPE parameter calculation supporting standard and YaRN.

    Args:
        head_dim: Attention head dimension.
        base: Base theta frequency.
        rope_type: Type of RoPE ("default", "yarn").
        rope_parameters: Dictionary of hyperparameters from Hugging Face configs.
        device: Target tensor device.

    Returns:
        inv_freq: Frequency tensor of shape (head_dim // 2,).
        attention_scaling: Scalar float factor for attention magnitude.
    """
    params = rope_parameters or {}
    rope_type = params.get("rope_type", rope_type).lower()

    if rope_type == "yarn":
        return compute_yarn_parameters(
            head_dim=head_dim,
            base=float(params.get("rope_theta", base)),
            factor=float(params.get("factor", 16.0)),
            original_max_position_embeddings=int(params.get("original_max_position_embeddings", 16384)),
            beta_fast=float(params.get("beta_fast", 32.0)),
            beta_slow=float(params.get("beta_slow", 1.0)),
            mscale=float(params.get("mscale", 1.0)),
            mscale_all_dim=float(params.get("mscale_all_dim", 1.0)),
            device=device,
        )

    # Standard RoPE fallback (Gemma, LLaMA-1/2)
    pos_freqs = base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    return 1.0 / pos_freqs, 1.0