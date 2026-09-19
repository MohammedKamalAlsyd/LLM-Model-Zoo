"""SigLIP Vision Tower with bicubic position interpolation and HF weight compatibility."""

import torch
import torch.nn.functional as F
from torch import nn
from configs import SigLipVisionConfig
from Zoo.Common.vision_utils import interpolate_2d_pos_embed

class SigLipVisionEmbeddings(nn.Module):
    """Extracts non-overlapping 2D image patches and applies positional embeddings."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels=cfg.num_channels,
            out_channels=self.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer("positions", torch.arange(self.num_patches), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Projects image patches and adds positional embeddings.

        Args:
            pixel_values: (Batch, Channels, Height, Width) tensor.

        Returns:
            Tensor of shape (Batch, Num_Patches, Embed_Dim).
        """
        _, _, h, w = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patches = self.patch_embedding(pixel_values.to(dtype=target_dtype)).flatten(2).transpose(1, 2)

        if h != self.cfg.image_size or w != self.cfg.image_size:
            pos_emb = interpolate_2d_pos_embed(
                self.position_embedding.weight, h, w, self.patch_size
            )
        else:
            pos_emb = self.position_embedding(self.positions)

        return patches + pos_emb


class SigLipAttention(nn.Module):
    """Multi-Head Attention module for SigLIP encoder layers."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # Fast FlashAttention/SDPA replacing manual matmul/softmax/dropout
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(attn.transpose(1, 2).contiguous().view(b, s, -1))


class SigLipMLP(nn.Module):
    """Two-layer Feed-Forward Network with GeLU approximation."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLipEncoderLayer(nn.Module):
    """Transformer encoder block with pre-LayerNorm residual connections."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.self_attn = SigLipAttention(cfg)
        self.mlp = SigLipMLP(cfg)
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SigLipEncoder(nn.Module):
    """Transformer encoder containing the stack of encoder layers."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SigLipEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SigLipVisionTransformer(nn.Module):
    """Inner vision transformer keeping exact checkpoint keys."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.embeddings = SigLipVisionEmbeddings(cfg)
        self.encoder = SigLipEncoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        return self.post_layernorm(x)


class SigLipVisionModel(nn.Module):
    """Top-level Vision Tower matching the HF key path `vision_tower.vision_model.*`."""

    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.vision_model = SigLipVisionTransformer(cfg)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)