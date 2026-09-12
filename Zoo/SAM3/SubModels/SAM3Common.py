import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Coordinate & Bounding Box Utilities
# ============================================================================

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def concat_padded_sequences(seq1: torch.Tensor, mask1: torch.Tensor, seq2: torch.Tensor, mask2: torch.Tensor):
    batch_size, seq1_length, hidden_size = seq1.shape
    batch_size2, seq2_length, _ = seq2.shape

    actual_seq1_lengths = mask1.sum(dim=-1)
    actual_seq2_lengths = mask2.sum(dim=-1)
    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length

    concatenated_mask = torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) < final_lengths[:, None]
    concatenated_sequence = torch.zeros((batch_size, max_length, hidden_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:, :seq1_length, :] = seq1

    index = torch.arange(seq2_length, device=seq2.device)[None].repeat(batch_size, 1) + actual_seq1_lengths[:, None]
    concatenated_sequence = concatenated_sequence.scatter(1, index[:, :, None].expand(-1, -1, hidden_size), seq2)

    return concatenated_sequence, concatenated_mask

# ============================================================================
# Core Neural Network Blocks
# ============================================================================

class Sam3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = nn.GELU() if act == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.dropout(self.fc1(x))))


class Sam3DecoderMLP(nn.Module):
    """2 or 3-layer MLP matching checkpoint keys (layer1, layer2, optional layer3)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        if num_layers == 2:
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = None
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
    def __init__(self, hidden_size: int, num_heads: int):
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
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        b, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(b, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, k_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, k_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.hidden_size)
        return self.o_proj(out), None

# ============================================================================
# Rotary & Sine Position Embeddings
# ============================================================================

def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def apply_rotary_pos_emb_2d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_out = (q.float() * cos) + (rotate_pairwise(q.float()) * sin)
    k_out = (k.float() * cos) + (rotate_pairwise(k.float()) * sin)
    return q_out.type_as(q), k_out.type_as(k)


class Sam3ViTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, head_dim: int = 64, rope_theta: float = 10000.0):
        super().__init__()
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: (N, 2) -> (w_coords, h_coords)
        pos_expanded = position_ids[..., None].float()
        freqs = pos_expanded * self.inv_freq.float()
        cos = freqs.cos()
        sin = freqs.sin()

        freq_h, freq_w = cos[:, 1], cos[:, 0]
        cos_hw = torch.cat([freq_h, freq_w], dim=-1)[None, ...].repeat_interleave(2, dim=-1)

        freq_h, freq_w = sin[:, 1], sin[:, 0]
        sin_hw = torch.cat([freq_h, freq_w], dim=-1)[None, ...].repeat_interleave(2, dim=-1)

        return cos_hw, sin_hw


class Sam3SinePositionEmbedding(nn.Module):
    def __init__(self, num_position_features: int = 64, temperature: int = 10000, normalize: bool = False):
        super().__init__()
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def encode_1d_positions(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_embed, y_embed = x * self.scale, y * self.scale
        dim_t = torch.arange(self.num_position_features, device=x.device, dtype=x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_position_features)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        dim_t = torch.arange(self.num_position_features, device=boxes.device, dtype=boxes.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_position_features)

        x_embed = boxes[:, :, 0] * self.scale
        y_embed = boxes[:, :, 1] * self.scale
        w_embed = boxes[:, :, 2] * self.scale
        h_embed = boxes[:, :, 3] * self.scale

        px = torch.stack(((x_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (x_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
        py = torch.stack(((y_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (y_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
        pw = torch.stack(((w_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (w_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
        ph = torch.stack(((h_embed[:, :, None] / dim_t)[:, :, 0::2].sin(), (h_embed[:, :, None] / dim_t)[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((py, px, pw, ph), dim=2)

    def forward(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        batch_size, _, height, width = shape
        y_embed = torch.arange(1, height + 1, dtype=dtype, device=device)[None, :, None].expand(batch_size, height, width)
        x_embed = torch.arange(1, width + 1, dtype=dtype, device=device)[None, None, :].expand(batch_size, height, width)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_position_features, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_position_features)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)