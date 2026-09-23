"""Ministral-3 Language Model with YaRN RoPE, LLaMA-4 Query Scaling, and GQA SDPA."""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Ministral3.configs import Ministral3TextConfig
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.RoPE import apply_rotary_pos_emb
from Zoo.Common.attention_utils import (
    create_causal_mask,
    get_llama_4_attn_scale,
    repeat_kv,
)
from Zoo.Common.RoPE import (
    compute_rope_parameters
)


# ============================================================================
# RoPE & Normalization
# ============================================================================

class Ministral3RMSNorm(nn.Module):
    """RMSNorm computed in FP32 for numerical stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return (normed * self.weight).to(orig_dtype)


class Ministral3RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding with corrected YaRN context extension."""

    inv_freq: torch.Tensor

    def __init__(self, config: Ministral3TextConfig) -> None:
        super().__init__()
        self.config = config
        inv_freq, self.attention_scaling = compute_rope_parameters(
            head_dim=config.head_dim,
            base=float(config.rope_parameters.get("rope_theta", 1000000.0)),
            rope_parameters=config.rope_parameters,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.unsqueeze(-1).float() @ self.inv_freq[None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(dtype=x.dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype=x.dtype)
        return cos, sin


# ============================================================================
# Attention & Feed-Forward Layers
# ============================================================================

class Ministral3Attention(nn.Module):
    """Grouped-Query Attention with YaRN RoPE, LLaMA-4 Scaling, and PyTorch SDPA."""

    def __init__(self, config: Ministral3TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # LLaMA-4 style query log-scaling for long context
        beta = self.config.rope_parameters.get("llama_4_scaling_beta", 0.1)
        orig_max = self.config.rope_parameters.get("original_max_position_embeddings", 16384)
        q = q * get_llama_4_attn_scale(position_ids, beta, orig_max).to(q.dtype)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Fused SDPA execution
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attention_mask is None and s > 1),
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(attn_out)


class Ministral3MLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: Ministral3TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Ministral3DecoderLayer(nn.Module):
    """Transformer Decoder Block: Pre-Norm Attention + Pre-Norm SwiGLU MLP."""

    def __init__(self, config: Ministral3TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Ministral3Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Ministral3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Pre-Norm Self Attention
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + attn_out

        # Pre-Norm SwiGLU FFN
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normed)
        return hidden_states


# ============================================================================
# Backbone & Causal LM Models
# ============================================================================

class Ministral3Model(nn.Module):
    """Ministral-3 Backbone matching Hugging Face `language_model.model` 1:1."""

    def __init__(self, config: Ministral3TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            Ministral3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        """Forward pass for text embeddings or raw tokens."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None

        _ , seq_len, _ = inputs_embeds.shape
        past_len = past_key_values.num_items() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=inputs_embeds.device
            ).unsqueeze(0)

        causal_mask = None
        if seq_len > 1 or past_len > 0 or self.config.sliding_window is not None:
            causal_mask = create_causal_mask(
                seq_len=seq_len,
                past_length=past_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                sliding_window=self.config.sliding_window,
            )
            if attention_mask is not None:
                pad_mask = (1.0 - attention_mask[:, None, None, :].to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
                causal_mask = causal_mask + pad_mask

        pos_emb = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=pos_emb,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
            )

        return {
            "last_hidden_state": self.norm(hidden_states),
            "past_key_values": past_key_values,
        }


class Ministral3ForCausalLM(nn.Module):
    """Causal LM wrapper matching Hugging Face `language_model` 1:1."""

    def __init__(self, config: Ministral3TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Ministral3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        """Ties lm_head weights to embed_tokens."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        logits_to_keep: Union[int, slice] = 0,
    ) -> Dict[str, Any]:
        """Calculates next-token logits with memory-efficient tail slicing."""
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        hidden_states = outputs["last_hidden_state"]

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, slice):
            hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states)
        return {
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
        }