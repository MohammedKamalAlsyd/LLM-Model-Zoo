import torch
import torch.nn as nn
from Zoo.SAM3.SubModels.SAM3Common import (
    Sam3Attention,
    Sam3MLP,
    Sam3ViTRotaryEmbedding,
    Sam3SinePositionEmbedding,
    apply_rotary_pos_emb_2d,
)

# ============================================================================
# Windowing Utilities
# ============================================================================

def window_partition(x: torch.Tensor, window_size: int):
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = h + pad_h, w + pad_w
    x = x.view(b, padded_h // window_size, window_size, padded_w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (padded_h, padded_w)

def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]):
    padded_h, padded_w = pad_hw
    h, w = hw
    b = windows.shape[0] // (padded_h * padded_w // window_size // window_size)
    x = windows.view(b, padded_h // window_size, padded_w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, padded_h, padded_w, -1)
    return x[:, :h, :w, :].contiguous()

# ============================================================================
# Vision Transformer Components
# ============================================================================

class Sam3ViTPatchEmbeddings(nn.Module):
    def __init__(self, patch_size: int = 14, in_channels: int = 3, hidden_size: int = 1024):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Sam3ViTEmbeddings(nn.Module):
    def __init__(self, patch_size: int = 14, hidden_size: int = 1024, pretrain_patches: int = 576):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embeddings = Sam3ViTPatchEmbeddings(patch_size=patch_size, hidden_size=hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, pretrain_patches, hidden_size))

    def _tile_position_embeddings(self, h_patches: int, w_patches: int) -> torch.Tensor:
        pretrain_dim = int(self.position_embeddings.shape[1] ** 0.5)
        hidden_size = self.position_embeddings.shape[-1]
        if pretrain_dim == h_patches and pretrain_dim == w_patches:
            return self.position_embeddings

        pos_embed = self.position_embeddings.reshape(1, pretrain_dim, pretrain_dim, hidden_size).permute(0, 3, 1, 2)
        repeat_h = h_patches // pretrain_dim + 1
        repeat_w = w_patches // pretrain_dim + 1
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[:, :, :h_patches, :w_patches]
        return pos_embed.permute(0, 2, 3, 1).reshape(1, h_patches * w_patches, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[-2:]
        embeddings = self.patch_embeddings(pixel_values)
        pos_embed = self._tile_position_embeddings(h // self.patch_size, w // self.patch_size)
        return embeddings + pos_embed


class Sam3ViTRoPEAttention(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        b, h, w, _ = x.shape
        seq_len = h * w
        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_2d(q, k, cos, sin)

        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, h, w, -1).contiguous()
        return self.o_proj(out)


class Sam3ViTLayer(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4736, num_heads: int = 16, window_size: int = 0):
        super().__init__()
        self.window_size = window_size
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.rotary_emb = Sam3ViTRotaryEmbedding(head_dim=hidden_size // num_heads)
        self.attention = Sam3ViTRoPEAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Sam3MLP(hidden_size, intermediate_size, act="gelu")

        # Fixed grid positions for RoPE
        hw_grid = (window_size, window_size) if window_size > 0 else (72, 72)
        hpos, wpos = torch.meshgrid(torch.arange(hw_grid[0]), torch.arange(hw_grid[1]), indexing="ij")
        pos_ids = torch.stack([wpos.flatten(), hpos.flatten()], dim=-1) * (24 / hw_grid[1])
        self.register_buffer("position_ids", pos_ids, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        if self.window_size > 0:
            h, w = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = window_partition(hidden_states, self.window_size)

        pos_embed = self.rotary_emb(hidden_states, self.position_ids)
        hidden_states = self.attention(hidden_states, pos_embed)

        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_hw, (h, w))

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Sam3ViTModel(nn.Module):
    """ViT Backbone holding `embeddings`, `layer_norm`, and `layers.0..31`."""
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4736, num_layers: int = 32, num_heads: int = 16):
        super().__init__()
        self.embeddings = Sam3ViTEmbeddings(patch_size=14, hidden_size=hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        global_indexes = {7, 15, 23, 31}
        self.layers = nn.ModuleList([
            Sam3ViTLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                window_size=0 if i in global_indexes else 24
            )
            for i in range(num_layers)
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        b = hidden_states.shape[0]
        h = pixel_values.shape[-2] // 14
        w = pixel_values.shape[-1] // 14

        hidden_states = hidden_states.view(b, h, w, -1)
        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states.view(b, h * w, -1)

# ============================================================================
# FPN Neck & Full Vision Backbone
# ============================================================================

class Sam3FPNLayer(nn.Module):
    def __init__(self, in_channels: int, fpn_dim: int, scale_factor: float):
        super().__init__()
        self.scale_layers = nn.ModuleList()

        if scale_factor == 4.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            self.scale_layers.append(nn.GELU())
            self.scale_layers.append(nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2))
            mid_c = in_channels // 4
        elif scale_factor == 2.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            mid_c = in_channels // 2
        elif scale_factor == 1.0:
            mid_c = in_channels
        elif scale_factor == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            mid_c = in_channels
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

        self.proj1 = nn.Conv2d(mid_c, fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.scale_layers:
            x = layer(x)
        return self.proj2(self.proj1(x))


class Sam3VisionNeck(nn.Module):
    def __init__(self, in_channels: int = 1024, fpn_dim: int = 256):
        super().__init__()
        self.position_encoding = Sam3SinePositionEmbedding(num_position_features=fpn_dim // 2, normalize=True)
        self.fpn_layers = nn.ModuleList([
            Sam3FPNLayer(in_channels=in_channels, fpn_dim=fpn_dim, scale_factor=scale)
            for scale in [4.0, 2.0, 1.0, 0.5]
        ])

    def forward(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        fpn_features, fpn_pos = (), ()
        for layer in self.fpn_layers:
            feat = layer(x)
            fpn_features += (feat,)
            fpn_pos += (self.position_encoding(feat.shape, feat.device, feat.dtype),)
        return fpn_features, fpn_pos


class Sam3VisionModel(nn.Module):
    """Comprises `backbone` (ViT) and `neck` (FPN). Exactly matches checkpoint hierarchy."""
    def __init__(self):
        super().__init__()
        self.backbone = Sam3ViTModel()
        self.neck = Sam3VisionNeck()

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.backbone(pixel_values)
        b = hidden_states.shape[0]
        h = pixel_values.shape[-2] // 14
        w = pixel_values.shape[-1] // 14

        spatial = hidden_states.view(b, h, w, -1).permute(0, 3, 1, 2)
        fpn_hidden_states, fpn_position_encoding = self.neck(spatial)

        return fpn_hidden_states, fpn_position_encoding