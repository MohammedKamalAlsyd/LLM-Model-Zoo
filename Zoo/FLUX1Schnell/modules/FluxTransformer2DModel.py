"""12B-Parameter Multimodal Diffusion Transformer (MMDiT) for FLUX."""

import math
from typing import NamedTuple, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.FLUX1Schnell.configs import FluxTransformerConfig


class Transformer2DModelOutput(NamedTuple):
    """Output container for the Transformer backbone."""
    sample: torch.Tensor


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization Zero for Dual-Stream MMDiT Blocks."""

    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Adaptive Layer Normalization Zero for Single-Stream DiT Blocks."""

    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    """Continuous Adaptive Layer Normalization for final output projection."""

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :] * scale
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines sinusoidal diffusion timesteps with pooled prompt projections."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.time_embedder = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.text_embedder = nn.Sequential(
            nn.Linear(pooled_projection_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        time_proj = get_timestep_embedding(timestep, 256, flip_sin_to_cos=True)
        time_emb = self.time_embedder(time_proj.to(dtype=pooled_projection.dtype))
        pooled_emb = self.text_embedder(pooled_projection)
        return time_emb + pooled_emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    freqs_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos, freqs)
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
    return freqs_cos, freqs_sin


def apply_rotary_emb(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs_cis
    cos = cos[None, :, None, :].to(x.device)
    sin = sin[None, :, None, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class FluxPosEmbed(nn.Module):
    """3D Multi-Axis RoPE: axis 0 (time), axis 1 (height), axis 2 (width)."""

    def __init__(self, theta: int = 10000, axes_dim: Tuple[int, int, int] = (16, 56, 56)) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_out, sin_out = [], []
        pos = ids.float()
        freqs_dtype = torch.float64 if ids.device.type != "mps" else torch.float32

        for i in range(ids.shape[-1]):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FeedForward(nn.Module):
    """FeedForward layer matching `ff.net.0.proj` and `ff.net.2` weights."""

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FluxAttention(nn.Module):
    """Multi-Head Attention supporting both Dual-Stream MMDiT and Single-Stream DiT."""

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 128,
        heads: int = 24,
        added_kv_proj_dim: Optional[int] = None,
        eps: float = 1e-6,
        pre_only: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = dim_head
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.pre_only = pre_only

        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=True)

        if not self.pre_only:
            self.to_out = nn.Sequential(
                nn.Linear(self.inner_dim, query_dim, bias=True),
                nn.Dropout(0.0),
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            enc_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.head_dim))
            enc_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.head_dim))
            enc_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.head_dim))

            enc_query = self.norm_added_q(enc_query)
            enc_key = self.norm_added_k(enc_key)

            query = torch.cat([enc_query, query], dim=1)
            key = torch.cat([enc_key, key], dim=1)
            value = torch.cat([enc_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # PyTorch Native SDPA
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            enc_out, hidden_out = out[:, :enc_len], out[:, enc_len:]
            hidden_out = self.to_out(hidden_out.contiguous())
            enc_out = self.to_add_out(enc_out.contiguous())
            return hidden_out, enc_out

        return out


class FluxTransformerBlock(nn.Module):
    """Dual-Stream MMDiT Block for joint image and text contextual processing."""

    def __init__(self, dim: int, num_attention_heads: int = 24, attention_head_dim: int = 128) -> None:
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_enc_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        attn_out, ctx_attn_out = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_enc_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out
        norm_hidden = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * ctx_attn_out
        norm_enc = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_enc)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-Stream DiT Block with joint linear projection and parallel attention/MLP."""

    def __init__(self, dim: int, num_attention_heads: int = 24, attention_head_dim: int = 128, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        x = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = x
        norm_x, gate = self.norm(x, emb=temb)
        mlp_out = self.act_mlp(self.proj_mlp(norm_x))
        attn_out = self.attn(hidden_states=norm_x, image_rotary_emb=image_rotary_emb)

        fused = torch.cat([attn_out, mlp_out], dim=2)
        x = residual + gate.unsqueeze(1) * self.proj_out(fused)

        return x[:, :text_seq_len], x[:, text_seq_len:]


class FluxTransformer2DModel(nn.Module):
    """12B-parameter FLUX Transformer combining Dual-stream and Single-stream DiT blocks."""

    def __init__(self, cfg: Optional[FluxTransformerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or FluxTransformerConfig()
        self.inner_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=self.cfg.theta, axes_dim=self.cfg.axes_dims_rope)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.cfg.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.cfg.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.cfg.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.cfg.num_attention_heads,
                attention_head_dim=self.cfg.attention_head_dim,
            )
            for _ in range(self.cfg.num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.cfg.num_attention_heads,
                attention_head_dim=self.cfg.attention_head_dim,
            )
            for _ in range(self.cfg.num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.cfg.in_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Scale timestep to [0, 1000] continuous boundary
        temb = self.time_text_embed(timestep * 1000.0, pooled_projections)

        # 3D RoPE concatenation
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # 19 Dual-Stream Blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # 38 Single-Stream Blocks
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return Transformer2DModelOutput(sample=output) if return_dict else output