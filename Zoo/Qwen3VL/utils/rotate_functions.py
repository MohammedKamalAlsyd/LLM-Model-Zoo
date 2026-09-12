from collections.abc import Sequence
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary embedding to queries and keys."""
    orig_dtype = q.dtype
    q = q.float()
    k = k.float()
    cos = cos.float()
    sin = sin.float()
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """Axial 2D RoPE for Qwen3-VL Vision Encoder."""
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Slicing avoids dynamic buffer .view() call issues in Pylance
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq[None, None, :]
        cos = freqs.cos()
        sin = freqs.sin()

        cos_hw = torch.cat([cos[:, 0], cos[:, 1]], dim=-1)
        sin_hw = torch.cat([sin[:, 0], sin[:, 1]], dim=-1)

        cos = torch.cat([cos_hw, cos_hw], dim=-1).unsqueeze(1)
        sin = torch.cat([sin_hw, sin_hw], dim=-1).unsqueeze(1)
        return cos, sin


class TextMRotaryEmbedding(nn.Module):
    """3D Multimodal RoPE (M-RoPE) for Qwen3-VL Language Model."""
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int = 128,
        theta: float = 5000000.0,
        mrope_section: Sequence[int] = (24, 20, 20),
    ):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.mrope_section = list(mrope_section)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (3, bs, seq_len)
        Returns: cos, sin of shape (bs, 1, seq_len, head_dim)
        """
        # Outer product via element-wise broadcasting:
        # (3, bs, seq_len, 1) * (1, 1, 1, head_dim // 2) -> (3, bs, seq_len, head_dim // 2)
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq[None, None, None, :]

        freqs_t = freqs[0]  # Base Temporal channel
        for dim, offset in enumerate((1, 2), start=1):  # 1=H, 2=W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        freqs_thw = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = freqs_thw.cos().unsqueeze(1)
        sin = freqs_thw.sin().unsqueeze(1)
        return cos, sin