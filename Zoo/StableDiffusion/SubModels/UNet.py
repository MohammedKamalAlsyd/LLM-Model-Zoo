import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

# =================================================================================
# Foundational Blocks
# =================================================================================


class TimestepEmbedSequential(nn.Module):
    """
    A block that dynamically routes time embeddings and context (text embeddings)
    to the specific layers that require them.
    """

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.children():
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# =================================================================================
# Attention & Transformer Blocks
# =================================================================================


class CrossAttention(nn.Module):
    """
    Handles both Self-Attention (if context is None) and Cross-Attention.
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        # SD uniquely uses bias=False for QKV projections in the UNet Attention
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # out_proj is sequential with a Linear layer (index 0) to match keys like 'to_out.0.weight'
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim))
        self.heads = heads
        self.dim_head = dim_head

    def forward(self, x, context=None):
        # If context is not provided, it falls back to self-attention
        context = context if context is not None else x

        batch_size, seq_len, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for SDPA: (Batch, Seq, Heads, Head_Dim) -> (Batch, Heads, Seq, Head_Dim)
        q = q.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # (Batch, Heads, Seq, Head_Dim) -> (Batch, Seq, Inner_Dim)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.heads * self.dim_head)
        )

        return self.to_out(out)


class GEGLU(nn.Module):
    """
    Variant of the GLU activation function used inside the Transformer.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class BasicTransformerBlock(nn.Module):
    """
    A Transformer Block composed of Self-Attention, Cross-Attention, and a Feed-Forward Network.
    """

    def __init__(self, dim, n_heads, d_head, context_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head
        )  # Self-Attention

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head
        )  # Cross-Attention

        self.norm3 = nn.LayerNorm(dim)

        # Wrapped in ModuleDict and Sequential to perfectly match:
        # transformer_blocks.0.ff.net.0.proj.weight
        # transformer_blocks.0.ff.net.2.weight
        self.ff = nn.ModuleDict(
            {
                "net": nn.Sequential(
                    GEGLU(dim, dim * 4),  # Index 0
                    nn.Identity(),  # Index 1 (Placeholder for Dropout in original repo)
                    nn.Linear(dim * 4, dim),  # Index 2
                )
            }
        )

    def forward(self, x, context=None):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff["net"](self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """
    Transforms spatial dimensions (Image/Latent HxW) into sequence dimensions,
    applies the Transformer blocks, and turns it back into spatial features.
    """

    def __init__(self, channels, n_heads, d_head, context_dim=768):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # SD v1.5 always defaults to 1 transformer block per SpatialTransformer
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, d_head, context_dim)]
        )

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context=None):
        batch, channels, height, width = x.shape
        residue = x

        x = self.norm(x)
        x = self.proj_in(x)

        # (Batch, Channels, Height, Width) -> (Batch, Channels, Seq) -> (Batch, Seq, Channels)
        x = x.view(batch, channels, height * width).transpose(1, 2)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        # (Batch, Seq, Channels) -> (Batch, Channels, Seq) -> (Batch, Channels, Height, Width)
        x = x.transpose(1, 2).contiguous().view(batch, channels, height, width)

        return self.proj_out(x) + residue


# =================================================================================
# Convolutional ResNet Block
# =================================================================================


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=1280):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(time_channels, out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels, eps=1e-5),
            nn.SiLU(),
            nn.Identity(),  # Dropout placeholder (index 2)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x, emb):
        h = self.in_layers(x)

        # Inject the time embeddings
        emb_out = self.emb_layers(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out

        h = self.out_layers(h)
        return self.skip_connection(x) + h


# =================================================================================
# Top-Level UNet Wrapper
# =================================================================================


class UNetModel(nn.Module):
    """
    The Main UNet Model predicting the Noise.

    Because the internal attributes strictly follow the `model.diffusion_model`
    naming conventions from original CompVis/StabilityAI code, you can directly
    load weights natively.

    Loading Example:
        state_dict = torch.load("v1-5-pruned.ckpt")["state_dict"]
        unet_dict = {
            k.replace("model.diffusion_model.", ""): v
            for k, v in state_dict.items() if k.startswith("model.diffusion_model.")
        }
        unet = UNetModel()
        unet.load_state_dict(unet_dict)
    """

    def __init__(self):
        super().__init__()

        # Matches model.diffusion_model.time_embed.* keys
        self.time_embed = nn.Sequential(
            nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280)
        )

        # Matches model.diffusion_model.input_blocks.* keys
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                TimestepEmbedSequential(
                    ResBlock(320, 320), SpatialTransformer(320, 8, 40)
                ),
                TimestepEmbedSequential(
                    ResBlock(320, 320), SpatialTransformer(320, 8, 40)
                ),
                TimestepEmbedSequential(Downsample(320)),
                TimestepEmbedSequential(
                    ResBlock(320, 640), SpatialTransformer(640, 8, 80)
                ),
                TimestepEmbedSequential(
                    ResBlock(640, 640), SpatialTransformer(640, 8, 80)
                ),
                TimestepEmbedSequential(Downsample(640)),
                TimestepEmbedSequential(
                    ResBlock(640, 1280), SpatialTransformer(1280, 8, 160)
                ),
                TimestepEmbedSequential(
                    ResBlock(1280, 1280), SpatialTransformer(1280, 8, 160)
                ),
                TimestepEmbedSequential(Downsample(1280)),
                TimestepEmbedSequential(ResBlock(1280, 1280)),
                TimestepEmbedSequential(ResBlock(1280, 1280)),
            ]
        )

        # Matches model.diffusion_model.middle_block.* keys
        self.middle_block = TimestepEmbedSequential(
            ResBlock(1280, 1280),
            SpatialTransformer(1280, 8, 160),
            ResBlock(1280, 1280),
        )

        # Matches model.diffusion_model.output_blocks.* keys
        self.output_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(ResBlock(2560, 1280)),
                TimestepEmbedSequential(ResBlock(2560, 1280)),
                TimestepEmbedSequential(ResBlock(2560, 1280), Upsample(1280)),
                TimestepEmbedSequential(
                    ResBlock(2560, 1280), SpatialTransformer(1280, 8, 160)
                ),
                TimestepEmbedSequential(
                    ResBlock(2560, 1280), SpatialTransformer(1280, 8, 160)
                ),
                TimestepEmbedSequential(
                    ResBlock(1920, 1280),
                    SpatialTransformer(1280, 8, 160),
                    Upsample(1280),
                ),
                TimestepEmbedSequential(
                    ResBlock(1920, 640), SpatialTransformer(640, 8, 80)
                ),
                TimestepEmbedSequential(
                    ResBlock(1280, 640), SpatialTransformer(640, 8, 80)
                ),
                TimestepEmbedSequential(
                    ResBlock(960, 640), SpatialTransformer(640, 8, 80), Upsample(640)
                ),
                TimestepEmbedSequential(
                    ResBlock(960, 320), SpatialTransformer(320, 8, 40)
                ),
                TimestepEmbedSequential(
                    ResBlock(640, 320), SpatialTransformer(320, 8, 40)
                ),
                TimestepEmbedSequential(
                    ResBlock(640, 320), SpatialTransformer(320, 8, 40)
                ),
            ]
        )

        # Matches model.diffusion_model.out.* keys
        self.out = nn.Sequential(
            nn.GroupNorm(32, 320, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )

    def forward(self, latent, context, time):
        """
        latent: (Batch_Size, 4, Height / 8, Width / 8)
        context: (Batch_Size, Seq_Len, Dim) -> Text Embeddings from CLIP
        time: (1, 320) -> Raw sinusoidal time embedding
        """
        # (1, 320) -> (1, 1280)
        t_emb = self.time_embed(time)

        skip_connections = []
        x = latent

        # Pass through the Encoder
        for module in self.input_blocks:
            x = module(x, t_emb, context)
            skip_connections.append(x)

        # Pass through Bottleneck
        x = self.middle_block(x, t_emb, context)

        # Pass through the Decoder
        for module in self.output_blocks:
            # Pop the matching skip connection from the encoder
            skip = skip_connections.pop()

            # Concatenate over the Channel dimension
            x = torch.cat((x, skip), dim=1)
            x = module(x, t_emb, context)

        # Final projection back to 4 Latent Channels
        return self.out(x)
