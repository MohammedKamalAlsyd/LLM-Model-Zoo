# Copyright 2024-2025 The Alibaba Wan Team Authors and Project Contributors.
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

__all__ = ["WanModel", "VaceWanModel", "WanUNet", "WanAttentionBlock", "Head", "MLPProj"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@amp.autocast(enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@amp.autocast(enabled=False)
def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int] = (-1, -1), qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs).transpose(1, 2)
        k = rope_apply(k, grid_sizes, freqs).transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(2)
        return self.o(out)


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(b, -1, n, d).transpose(1, 2)
        v = self.v(context).view(b, -1, n, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(2)
        return self.o(out)


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim: int, num_heads: int, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        img_len = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :img_len]
        context_txt = context[:, img_len:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d).transpose(1, 2)

        k_txt = self.norm_k(self.k(context_txt)).view(b, -1, n, d).transpose(1, 2)
        v_txt = self.v(context_txt).view(b, -1, n, d).transpose(1, 2)
        out_txt = F.scaled_dot_product_attention(q, k_txt, v_txt).transpose(1, 2).flatten(2)

        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d).transpose(1, 2)
        v_img = self.v_img(context_img).view(b, -1, n, d).transpose(1, 2)
        out_img = F.scaled_dot_product_attention(q, k_img, v_img).transpose(1, 2).flatten(2)

        return self.o(out_txt + out_img)


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with amp.autocast(dtype=torch.float32):
            e_mod = (self.modulation + e).chunk(6, dim=1)

        y = self.self_attn(self.norm1(x).float() * (1 + e_mod[1]) + e_mod[0], seq_lens, grid_sizes, freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e_mod[2]

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e_mod[4]) + e_mod[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * e_mod[5]
        return x


class Head(nn.Module):

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        with amp.autocast(dtype=torch.float32):
            e_mod = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e_mod[1]) + e_mod[0])
        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, flf_pos_emb: bool = False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, in_dim))

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "emb_pos"):
            b, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d) + self.emb_pos
        return self.proj(image_embeds)


class WanModel(ModelMixin, ConfigMixin):
    """
    Unified 3D Diffusion Transformer Backbone for Wan2.1.
    Directly compatible with safetensors weights from Wan-AI on HuggingFace.
    """

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type in ("i2v", "flf2v"):
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=(model_type == "flf2v"))

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clip_fea is not None and hasattr(self, "img_emb"):
            context_clip = self.img_emb(clip_fea)
            context = torch.cat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=None,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.head(x, e)
        return [u.float() for u in self.unpatchify(x, grid_sizes)]

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> List[torch.Tensor]:
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out


class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self, *args, block_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c: torch.Tensor, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        return c, self.after_proj(c)


class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self, *args, block_id: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_id = block_id

    def forward(self, x: torch.Tensor, hints: List[torch.Tensor], context_scale: float = 1.0, **kwargs) -> torch.Tensor:
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanModel(WanModel):

    def __init__(self, *args, vace_layers: Optional[List[int]] = None, vace_in_dim: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vace_layers = [i for i in range(0, len(self.blocks), 2)] if vace_layers is None else vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                "t2v_cross_attn",
                self.dim,
                self.config.ffn_dim if hasattr(self, "config") else 13824,
                self.num_heads,
                block_id=self.vace_layers_mapping.get(i, None),
            )
            for i in range(len(self.blocks))
        ])

        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                "t2v_cross_attn",
                self.dim,
                self.config.ffn_dim if hasattr(self, "config") else 13824,
                self.num_heads,
                block_id=i,
            )
            for i in self.vace_layers
        ])
        self.vace_patch_embedding = nn.Conv3d(
            vace_in_dim or 16, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )


# Alias UNet to WanModel for standard diffusers/SD project nomenclature
WanUNet = WanModel