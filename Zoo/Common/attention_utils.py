"""Universal attention, head repetition, and masking utilities for LLMs."""

from typing import Optional
import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands Key/Value heads to match Query head count in Grouped-Query Attention (GQA).

    Args:
        hidden_states: KV tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        n_rep: Number of times each KV head is repeated (num_heads // num_kv_heads).

    Returns:
        Tensor of shape (batch, num_heads, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states
    b, n_kv, s, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


def get_llama_4_attn_scale(
    position_ids: torch.Tensor,
    beta: float = 0.1,
    original_max_position_embeddings: int = 16384,
) -> torch.Tensor:
    """Computes log-frequency query scaling for long-context attention stability.

    Args:
        position_ids: Absolute token positions of shape (batch, seq_len).
        beta: Multiplicative scaling hyperparameter.
        original_max_position_embeddings: Pre-trained context ceiling.

    Returns:
        Tensor of shape (batch, 1, seq_len, 1) broadcastable over heads and head_dim.
    """
    bucket_index = torch.floor(position_ids.float() / original_max_position_embeddings)
    scaling = 1.0 + beta * torch.log(1.0 + bucket_index)
    return scaling[:, None, :, None]


def create_causal_mask(
    seq_len: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Creates an additive causal attention mask with optional sliding-window constraint.

    Args:
        seq_len: Length of the current input tokens.
        past_length: Number of past tokens already cached in the KV cache.
        dtype: Target floating point precision for negative infinity.
        device: Device where the mask is allocated.
        sliding_window: Optional maximum token lookback distance.

    Returns:
        Tensor of shape (1, 1, seq_len, past_length + seq_len).
    """
    kv_len = past_length + seq_len
    neg_inf = torch.finfo(dtype).min

    mask = torch.full((seq_len, kv_len), fill_value=neg_inf, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1 + past_length)

    if sliding_window is not None:
        q_pos = torch.arange(seq_len, device=device)[:, None] + past_length
        k_pos = torch.arange(kv_len, device=device)[None, :]
        mask = torch.where((q_pos - k_pos) <= sliding_window, mask, neg_inf)

    return mask.unsqueeze(0).unsqueeze(0)