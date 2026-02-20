"""
Minimal Mistral3 building blocks used in this repository.

This file provides small helper dataclasses and modules used by the
Mistral3 model implementation: rotary embedding helpers, attention,
MLP, RMSNorm and decoder layer. The goal of edits here is purely to
improve readability: added docstrings and inline comments while
preserving public attribute names so checkpoints (Hugging Face weights)
can be loaded without name changes.
"""

from typing import Optional,Dict, Any

import torch
import torch.nn as nn
from dataclasses import dataclass
from Pixtral import PixtralConfig
from Zoo.Mistral3.utils.KVCache import KVCache
from Zoo.Mistral3.utils.rotate_functions import apply_rotary_pos_emb


@dataclass
class RopeParameters:
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    factor: float = 16.0
    llama_4_scaling_beta: float = 0.1
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    original_max_position_embeddings: int = 16384
    rope_theta: float = 1000000.0
    rope_type: str = "yarn"
    type: str = "yarn"


@dataclass
class Ministral3Config:
    attention_dropout: float = 0.0
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 14336
    max_position_embeddings: int = 262144
    num_attention_heads: int = 32
    num_hidden_layers: int = 34
    num_key_value_heads: int = 8  # For GQA
    rms_norm_eps: float = 1e-5
    rope_parameters: dict = RopeParameters().__dict__
    vocab_size: int = 131072
    pad_token_id: int | None = 11
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2


@dataclass
class Mistral3Config:
    spatial_merge_size: int = 2
    text_config: Ministral3Config = Ministral3Config()
    vision_config: PixtralConfig = PixtralConfig()


# --------
# Helper Methods
# --------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Expand the middle dim to repeat key/value heads for GQA-style grouping
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )  # (batch, num_key_value_heads, n_rep, seqlen, head_dim)

    # Flatten the repeated dim into the attention-head dimension
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )  # (batch, num_attention_heads, seqlen, head_dim)


def _get_llama_4_attn_scale(
    positions_ids: torch.Tensor, beta: float, max_position_embeddings: int
) -> torch.Tensor:
    # Llama-4 style attention scaling based on position ranges. Positions are
    # bucketed into intervals of length `max_position_embeddings` and scaled
    # logarithmically with `beta`.
    scaling = 1 + beta * torch.log(
        1 + torch.floor(positions_ids / max_position_embeddings)
    )
    return scaling.unsqueeze(-1)


def create_causal_mask(
    config,
    inputs_embeds: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[KVCache] = None,
) -> Optional[torch.Tensor]:
    """
    Build an additive causal attention mask.

    Returns:
        Tensor of shape (batch, 1, query_length, key_value_length)
        containing:
            0        → allowed attention
            -inf     → masked attention

    Compatible with:
        attn_weights = attn_weights + attention_mask
    """

    if inputs_embeds is None:
        return None
    
    batch_size, query_length, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # -----------------------------------------------------
    # Determine total KV length (past + current)
    # -----------------------------------------------------
    past_length = 0
    if past_key_values is not None:
        past_length = past_key_values.num_items()

    kv_length = past_length + query_length

    # -----------------------------------------------------
    # Base causal mask
    # -----------------------------------------------------
    # Build (query_length, kv_length) matrix
    mask = torch.full(
        (query_length, kv_length),
        fill_value=torch.finfo(dtype).min,
        device=device,
    )

    # Allow attention to previous + current tokens
    mask = torch.triu(mask, diagonal=1 + past_length)

    # -----------------------------------------------------
    # Optional sliding window support
    # -----------------------------------------------------
    sliding_window = getattr(config.text_config, "sliding_window", None)

    if sliding_window is not None:
        # Create window constraint
        q_ids = torch.arange(query_length, device=device) + past_length
        kv_ids = torch.arange(kv_length, device=device)

        distance = q_ids[:, None] - kv_ids[None, :]
        window_mask = distance <= sliding_window

        # Combine with causal mask
        mask = torch.where(window_mask, mask, torch.finfo(dtype).min)

    # -----------------------------------------------------
    # Expand to 4D (batch, 1, q_len, kv_len)
    # -----------------------------------------------------
    mask = mask.unsqueeze(0).unsqueeze(1)
    mask = mask.expand(batch_size, 1, query_length, kv_length)

    # -----------------------------------------------------
    # Apply padding mask if provided
    # -----------------------------------------------------
    if attention_mask is not None:
        # Expected shape: (batch, kv_length)
        padding_mask = attention_mask[:, None, None, :]
        padding_mask = (1.0 - padding_mask) * torch.finfo(dtype).min
        mask = mask + padding_mask

    return mask


class Ministral3Attention(nn.Module):
    """Multi-headed attention module used by Ministral3.

    This module implements standard scaled dot-product attention with
    support for grouped key/value heads (GQA) via `num_key_value_heads`.
    Preserve projection attribute names (`q_proj`, `k_proj`, `v_proj`,
    `o_proj`) so that checkpoint loading remains compatible.
    """

    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def apply_attention(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: float = 1.0,
        dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expand grouped key/value heads to match query head count
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)

        # Compute scaled dot-product attention
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in fp32 for numeric stability, then cast back
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache_position: torch.LongTensor,
        past_key_values: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: (batch, seq_len, hidden_size)
        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_length)

        # target shape for projections: (batch, seq_len, num_heads, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project to q/k/v and move the heads to dim=1: -> (batch, num_heads, seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary positional embeddings (cos, sin) to queries and keys
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Optionally scale queries by Llama-4 attention scaling factor
        query_states = query_states * _get_llama_4_attn_scale(
            cache_position,
            self.config.rope_parameters.get("llama_4_scaling_beta", 0.1),
            self.config.rope_parameters.get("original_max_position_embeddings", 16384),
        ).to(query_states.dtype)

        # If we have cached key/values (autoregressive generation), update them
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # Compute attention and project the output back to hidden_size
        attn_output, attn_weights = self.apply_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            self.attention_dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Ministral3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Gated linear unit style MLP: up_proj * activation(gate_proj)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Ministral3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Ministral3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # scale by learned weight and restore original dtype
        return self.weight * hidden_states.to(input_dtype)


class Ministral3DecoderLayer(nn.Module):
    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Ministral3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(config)
        self.input_layernorm = Ministral3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Ministral3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: KVCache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Pre-norm residual block: input -> norm -> self-attention -> add residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention (returns attn output and optional weights)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Feed-forward block with post-attention norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Ministral3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def compute_default_rope_parameters(
        self, config: Ministral3Config
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this variant of RoPE

        # Compute inverse frequencies for rotary embeddings following the classic formula
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(
            self.config
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def forward(self, x, position_ids):
        # inv_freq: (dim/2,)
        # Build (batch, dim/2, seq_len) frequency matrix and then stack to match
        # (batch, seq_len, dim) after transposing.
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        # (batch, dim/2, seq_len) -> transpose -> (batch, seq_len, dim/2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )

        # Duplicate freqs to interleave for cos/sin pairs: (batch, seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Ministral3Model(nn.Module):
    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Ministral3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values = KVCache()

        if cache_position is None and inputs_embeds is not None:
            past_seen_tokens = (
                past_key_values.num_items() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values



class Ministral3ForCausalLM(nn.Module):
    """
    Minimal causal language modeling head on top of Ministral3Model.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = Ministral3Model(config)
        self.vocab_size = config.vocab_size

        # Output projection
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Tie weights (if desired)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "loss": optional,
                "logits": Tensor,
                "past_key_values": optional
            }
        """

        # -----------------------------------------------------
        # 1. Run base transformer
        # -----------------------------------------------------
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        hidden_states = outputs["last_hidden_state"]

        # -----------------------------------------------------
        # 2. Optionally slice logits (generation optimization)
        # -----------------------------------------------------
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]

        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": outputs.get("past_key_values", None),
        }