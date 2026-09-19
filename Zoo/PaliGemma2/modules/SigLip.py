"""SigLIP implementation maintaining exact HuggingFace weight compatibility."""

import torch
import torch.nn.functional as F
from torch import nn
from configs import SigLipVisionConfig

class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size, 
            kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.position_embedding = nn.Embedding(num_patches, cfg.hidden_size)
        self.register_buffer("positions", torch.arange(num_patches), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.positions)


class SigLipAttention(nn.Module):
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
    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLipEncoderLayer(nn.Module):
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



class SigLipVisionTransformer(nn.Module):
    """Preserves vision_tower.vision_model key path."""
    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.embeddings = SigLipVisionEmbeddings(cfg)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([SigLipEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        for layer in getattr(self.encoder, "layers"):
            x = layer(x)
        return self.post_layernorm(x)


class SigLipVisionModel(nn.Module):
    """Wrapper so checkpoints map to vision_tower.vision_model.*"""
    def __init__(self, cfg: SigLipVisionConfig) -> None:
        super().__init__()
        self.vision_model = SigLipVisionTransformer(cfg)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)