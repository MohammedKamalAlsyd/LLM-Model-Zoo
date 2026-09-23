"""Latent UNet noise prediction model with Spatial Cross-Attention matching official SD weights."""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.StableDiffusion.configs import UNetConfig


# =============================================================================
# Dynamic Sequential Containers & Spatial Resampling
# =============================================================================

class TimestepEmbedSequential(nn.Sequential):
    """Sequential container routing time embeddings and context tokens to receptive layers."""

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

            x: Latent features of shape (batch_size, channels, height, width).
            time_emb: Projected timestep embeddings of shape (batch_size, time_embed_dim).
            context: Text sequence embeddings from CLIP of shape (batch_size, seq_len, context_dim).

        Returns:
            Tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, time_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """Strided 2D convolution downsampling spatial dimensions by 2x."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbor interpolation followed by 2D convolution for 2x spatial upsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# =============================================================================
# Attention & Spatial Transformer Blocks
# =============================================================================

class CrossAttention(nn.Module):
    """Grouped attention supporting Self-Attention (context=None) and Cross-Attention."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        # Stable Diffusion v1.5 uses bias=False for Q, K, V projections in the UNet
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # Output projection wrapped in Sequential to match 'to_out.0.weight'
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim))
        self.heads = heads
        self.dim_head = dim_head

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:

            x: Flattened spatial query features of shape (batch, seq_len_q, query_dim).
            context: Optional text key/value states of shape (batch, seq_len_kv, context_dim).

        Returns:
            Tensor of shape (batch, seq_len_q, query_dim).
        """
        k_source = context if context is not None else x
        v_source = context if context is not None else x

        b, s_q, _ = x.shape
        _, s_kv, _ = k_source.shape

        q = self.to_q(x).view(b, s_q, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(k_source).view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(v_source).view(b, s_kv, self.heads, self.dim_head).transpose(1, 2)

        # FlashAttention / Native PyTorch SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, s_q, self.heads * self.dim_head)
        return self.to_out(out)


class GEGLU(nn.Module):
    """Gated Linear Unit activation with standard GeLU gating."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class BasicTransformerBlock(nn.Module):
    """Transformer block with Self-Attention, Cross-Attention, and GEGLU Feed-Forward."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int = 768,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head)

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head
        )

        self.norm3 = nn.LayerNorm(dim)
        # ModuleDict and Sequential precisely match 'transformer_blocks.0.ff.net.0.*'
        self.ff = nn.ModuleDict(
            {
                "net": nn.Sequential(
                    GEGLU(dim, dim * 4),
                    nn.Identity(),  # Placeholder for dropout (Index 1)
                    nn.Linear(dim * 4, dim),
                )
            }
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        x = x + self.attn1(self.norm1(x))
        # Pre-LN Text Cross-Attention
        x = x + self.attn2(self.norm2(x), context=context)
        # Pre-LN Feed-Forward
        x = x + self.ff["net"](self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """Maps spatial 2D feature grids to 1D token sequences, processes via Transformer, and unrolls."""

    def __init__(
        self,
        channels: int,
        n_heads: int,
        d_head: int,
        context_dim: int = 768,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, d_head, context_dim)]
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        x_norm = self.proj_in(self.norm(x))
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_seq = x_norm.view(b, c, h * w).transpose(1, 2)

        for block in self.transformer_blocks:
            x_seq = block(x_seq, context=context)

        # (B, H*W, C) -> (B, C, H, W)
        x_out = x_seq.transpose(1, 2).contiguous().view(b, c, h, w)
        return self.proj_out(x_out) + res


# =============================================================================
# Convolutional ResNet Residual Block
# =============================================================================

class ResBlock(nn.Module):
    """Residual block conditioning feature maps with injected sinusoidal time embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int = 1280,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels, eps=eps),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels, eps=eps),
            nn.SiLU(),
            nn.Identity(),  # Dropout placeholder (Index 2)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        # Broadcast time embedding across spatial dims: (B, C, 1, 1)
        time_proj = self.emb_layers(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_proj
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# =============================================================================
# Top-Level UNet Backbone
# =============================================================================

class UNetModel(nn.Module):
    """Conditional Latent UNet noise residual predictor matching `model.diffusion_model.*` keys."""

    def __init__(self, cfg: Optional[UNetConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or UNetConfig()
        time_dim = self.cfg.time_embed_dim
        ctx_dim = self.cfg.context_dim
        heads = self.cfg.num_heads

        # 1. Sinusoidal Time Embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(self.cfg.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 2. Downsampling Stages (input_blocks.0 to input_blocks.11)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(self.cfg.in_channels, 320, kernel_size=3, padding=1)),
                TimestepEmbedSequential(ResBlock(320, 320, time_dim), SpatialTransformer(320, heads, 40, ctx_dim)),
                TimestepEmbedSequential(ResBlock(320, 320, time_dim), SpatialTransformer(320, heads, 40, ctx_dim)),
                TimestepEmbedSequential(Downsample(320)),
                TimestepEmbedSequential(ResBlock(320, 640, time_dim), SpatialTransformer(640, heads, 80, ctx_dim)),
                TimestepEmbedSequential(ResBlock(640, 640, time_dim), SpatialTransformer(640, heads, 80, ctx_dim)),
                TimestepEmbedSequential(Downsample(640)),
                TimestepEmbedSequential(ResBlock(640, 1280, time_dim), SpatialTransformer(1280, heads, 160, ctx_dim)),
                TimestepEmbedSequential(ResBlock(1280, 1280, time_dim), SpatialTransformer(1280, heads, 160, ctx_dim)),
                TimestepEmbedSequential(Downsample(1280)),
                TimestepEmbedSequential(ResBlock(1280, 1280, time_dim)),
                TimestepEmbedSequential(ResBlock(1280, 1280, time_dim)),
            ]
        )

        # 3. Bottleneck Stage (middle_block)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(1280, 1280, time_dim),
            SpatialTransformer(1280, heads, 160, ctx_dim),
            ResBlock(1280, 1280, time_dim),
        )

        # 4. Upsampling Stages (output_blocks.0 to output_blocks.11)
        self.output_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(ResBlock(2560, 1280, time_dim)),
                TimestepEmbedSequential(ResBlock(2560, 1280, time_dim)),
                TimestepEmbedSequential(ResBlock(2560, 1280, time_dim), Upsample(1280)),
                TimestepEmbedSequential(ResBlock(2560, 1280, time_dim), SpatialTransformer(1280, heads, 160, ctx_dim)),
                TimestepEmbedSequential(ResBlock(2560, 1280, time_dim), SpatialTransformer(1280, heads, 160, ctx_dim)),
                TimestepEmbedSequential(ResBlock(1920, 1280, time_dim), SpatialTransformer(1280, heads, 160, ctx_dim), Upsample(1280)),
                TimestepEmbedSequential(ResBlock(1920, 640, time_dim), SpatialTransformer(640, heads, 80, ctx_dim)),
                TimestepEmbedSequential(ResBlock(1280, 640, time_dim), SpatialTransformer(640, heads, 80, ctx_dim)),
                TimestepEmbedSequential(ResBlock(960, 640, time_dim), SpatialTransformer(640, heads, 80, ctx_dim), Upsample(640)),
                TimestepEmbedSequential(ResBlock(960, 320, time_dim), SpatialTransformer(320, heads, 40, ctx_dim)),
                TimestepEmbedSequential(ResBlock(640, 320, time_dim), SpatialTransformer(320, heads, 40, ctx_dim)),
                TimestepEmbedSequential(ResBlock(640, 320, time_dim), SpatialTransformer(320, heads, 40, ctx_dim)),
            ]
        )

        # 5. Output Projection Head
        self.out = nn.Sequential(
            nn.GroupNorm(32, 320, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(320, self.cfg.out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

            latent: Latent tensor of shape (batch, 4, height, width).
            context: Text condition tokens of shape (batch, 77, 768).
            time_embedding: Raw sinusoidal embedding of shape (batch, 320) or (1, 320).

        Returns:
            Predicted noise residual tensor of shape (batch, 4, height, width).
        """
        t_emb = self.time_embed(time_embedding)
        skip_connections: List[torch.Tensor] = []
        x = latent

        # Encoder path
        for module in self.input_blocks:
            x = module(x, t_emb, context)
            skip_connections.append(x)

        # Middle bottleneck
        x = self.middle_block(x, t_emb, context)

        # Decoder path with skip concatenation
        for module in self.output_blocks:
            skip = skip_connections.pop()
            x = torch.cat((x, skip), dim=1)
            x = module(x, t_emb, context)

        return self.out(x)