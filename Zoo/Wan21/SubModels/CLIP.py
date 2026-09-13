# Copyright 2024-2025 The Alibaba Wan Team Authors and Project Contributors.
import logging
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

__all__ = ["XLMRobertaCLIP", "CLIPModel"]


def pos_interpolate(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    if pos.size(1) == seq_len:
        return pos
    src_grid = int(math.sqrt(pos.size(1)))
    tar_grid = int(math.sqrt(seq_len))
    n = pos.size(1) - src_grid * src_grid
    return torch.cat(
        [
            pos[:, :n],
            F.interpolate(
                pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2),
                size=(tar_grid, tar_grid),
                mode="bicubic",
                align_corners=False,
            )
            .flatten(2)
            .transpose(1, 2),
        ],
        dim=1,
    )


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c = x.shape
        q, k, v = (
            self.to_qkv(x)
            .view(b, s, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b, s, c)
        return self.proj(out)


class AttentionBlock(nn.Module):

    def __init__(self, dim: int, mlp_ratio: float, num_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        mid_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mid_dim),
            QuickGELU(),
            nn.Linear(mid_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        num_layers: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, dim) * (1.0 / math.sqrt(dim)))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim) * (1.0 / math.sqrt(dim)))
        self.pre_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, norm_eps) for _ in range(num_layers)
        ])
        self.post_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, use_31_block: bool = True) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embedding
        x = self.pre_norm(x)

        if use_31_block:
            return self.transformer[:-1](x)
        return self.transformer(x)


class XLMRobertaCLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer(
            image_size=224,
            patch_size=14,
            dim=1280,
            mlp_ratio=4.0,
            num_heads=16,
            num_layers=32,
        )


class CLIPModel:
    """
    Unified CLIP Visual Encoder Interface extracting Penultimate Layer (Block 31)
    representations (1 class token + 256 patch tokens = 257 tokens).
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        self.dtype = dtype
        self.device = torch.device(device)

        self.model = XLMRobertaCLIP().to(dtype=dtype, device=self.device).eval().requires_grad_(False)

        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"Loading CLIP visual weights from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)

        self.transforms = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device="cuda", dtype=torch.float16):
        pth = os.path.join(checkpoint_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        return cls(checkpoint_path=pth, device=device, dtype=dtype)

    def visual(self, videos: List[torch.Tensor]) -> torch.Tensor:
        """
        Accepts list of tensors each of shape [C=3, 1, H, W] in range [-1, 1].
        Returns token tensor of shape [B, 257 * num_frames, 1280].
        """
        size = (224, 224)
        processed = torch.cat([
            F.interpolate(u.transpose(0, 1), size=size, mode="bicubic", align_corners=False)
            for u in videos
        ])
        processed = self.transforms(processed.mul(0.5).add(0.5))

        with torch.cuda.amp.autocast(dtype=self.dtype):
            return self.model.visual(processed, use_31_block=True)