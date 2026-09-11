import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Model Output Container
# ==============================================================================

class Transformer2DModelOutput(NamedTuple):
    sample: torch.Tensor


# ==============================================================================
# Normalization Layers (AdaLN)
# ==============================================================================

class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization Zero for Dual-Stream MMDiT Blocks."""

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Adaptive Layer Normalization Zero for Single-Stream DiT Blocks."""

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    """Continuous Adaptive Layer Normalization for the output projection."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        bias: bool = True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ==============================================================================
# Embeddings & Rotary Position Embedding (RoPE)
# ==============================================================================

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps must be a 1D tensor."
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :] * scale
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.act_1 = nn.SiLU() if act_fn == "silu" else nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(caption)))


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    freqs_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos, freqs)
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
    return freqs_cos, freqs_sin


def apply_rotary_emb(x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs_cis
    cos = cos[None, :, None, :].to(x.device)
    sin = sin[None, :, None, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int = 10000, axes_dim: tuple[int, int, int] = (16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out, sin_out = [], []
        pos = ids.float()
        freqs_dtype = torch.float64 if ids.device.type != "mps" else torch.float32

        for i in range(n_axes):
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


# ==============================================================================
# FeedForward & Attention Modules
# ==============================================================================

class GELUApproximate(nn.Module):
    """Matches diffusers `ff.net.0.proj` state_dict key structure."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    """FeedForward layer matching `ff.net.0.proj` and `ff.net.2` weights."""

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([
            GELUApproximate(dim, inner_dim, bias=bias),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim_out, bias=bias),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class FluxAttention(nn.Module):
    """Multi-Head Attention supporting both Dual-stream (MMDiT) and Single-stream DiT."""

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 128,
        heads: int = 24,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        out_bias: bool = True,
        eps: float = 1e-6,
        pre_only: bool = False,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.pre_only = pre_only

        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                nn.Linear(self.inner_dim, query_dim, bias=out_bias),
                nn.Dropout(0.0),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

        # High-performance PyTorch 2.0+ Scaled Dot-Product Attention
        q = query.transpose(1, 2)  # [B, H, S, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            enc_out, hidden_out = hidden_states[:, :enc_len], hidden_states[:, enc_len:]
            hidden_out = self.to_out[0](hidden_out.contiguous())
            hidden_out = self.to_out[1](hidden_out)
            enc_out = self.to_add_out(enc_out.contiguous())
            return hidden_out, enc_out

        return hidden_states


# ==============================================================================
# Transformer Blocks
# ==============================================================================

class FluxTransformerBlock(nn.Module):
    """Dual-stream MMDiT Block processing text and image branches jointly."""

    def __init__(self, dim: int, num_attention_heads: int = 24, attention_head_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=True,
            eps=eps,
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
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Image branch
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)

        # Text/Context branch
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = (
            self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(
            norm_encoder_hidden_states
        )

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT Block with parallel attention and MLP computation."""

    def __init__(
        self, dim: int, num_attention_heads: int = 24, attention_head_dim: int = 128, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=True,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = residual + gate.unsqueeze(1) * self.proj_out(hidden_states)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


# ==============================================================================
# Full FluxTransformer2DModel
# ==============================================================================

class FluxTransformer2DModel(nn.Module):
    """
    Flux Transformer 2D Model (FLUX.1-schnell and FLUX.1-dev compatible).

    Matches configuration:
      - in_channels: 64
      - joint_attention_dim: 4096
      - pooled_projection_dim: 768
      - num_layers (dual-stream): 19
      - num_single_layers: 38
      - num_attention_heads: 24
      - attention_head_dim: 128
      - guidance_embeds: False (Schnell) / True (Dev)
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        if guidance_embeds:
            self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
            )
        else:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
            )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor, ...] | Transformer2DModelOutput:
        # 1. Project input patches and conditioning
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000.0
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000.0

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        # 2. Rotary Position Embeddings (RoPE)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # 3. Dual-Stream MMDiT Blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # 4. Single-Stream DiT Blocks
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # 5. Output Normalization & Linear Projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)