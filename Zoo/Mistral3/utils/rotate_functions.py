import torch


def rotate_half(hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input.

    Used in RoPE to apply rotation: (x1, x2) -> (-x2, x1)
    """
    x1 = hidden_state[..., : hidden_state.shape[-1] // 2]
    x2 = hidden_state[..., hidden_state.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
            - For [batch, heads, seq_len, head_dim] use unsqueeze_dim=1
            - For [batch, seq_len, heads, head_dim] use unsqueeze_dim=2

    Returns:
        Tuple of (q_rotated, k_rotated)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
