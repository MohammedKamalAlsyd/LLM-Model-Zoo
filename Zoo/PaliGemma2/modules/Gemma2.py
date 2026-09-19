"""Compact Gemma 2 language model preserving exact Hugging Face weight compatibility."""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from configs import Gemma2Config
from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.RoPE import apply_rotary_pos_emb


class Gemma2RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with Gemma 2 offset scaling (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        # Gemma 2 offset scaling: (1 + weight)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class Gemma2MLP(nn.Module):
    """Gated feed-forward network using approximate GeLU."""

    def __init__(self, cfg: Gemma2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma2RotaryEmbedding(nn.Module):
    """Calculates Rotary Positional Embeddings (RoPE) frequencies."""

    def __init__(self, cfg: Gemma2Config) -> None:
        super().__init__()
        dim = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pos_ids: (batch, seq_len)
        freqs = torch.einsum("bi,j->bij", pos_ids.float(), self.inv_freq.to(pos_ids.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Gemma2Attention(nn.Module):
    """Grouped Query Attention (GQA) with logit softcapping and corrected scaling."""

    def __init__(self, cfg: Gemma2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = self.num_heads // self.num_kv_heads

        scalar = cfg.query_pre_attn_scalar if cfg.query_pre_attn_scalar is not None else float(self.head_dim)
        self.scaling = scalar ** -0.5
        self.softcap = cfg.attn_logit_softcapping

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # Expand KV heads for Grouped Query Attention (GQA)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Scaled Dot-Product with Softcapping
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if self.softcap is not None:
            scores = torch.tanh(scores / self.softcap) * self.softcap

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)


class Gemma2DecoderLayer(nn.Module):
    """Gemma 2 block featuring 4 sandwich LayerNorms."""

    def __init__(self, cfg: Gemma2Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Gemma2Attention(cfg, layer_idx)
        self.mlp = Gemma2MLP(cfg)

        # 4 Sandwich LayerNorms required by Gemma 2
        self.input_layernorm = Gemma2RMSNorm(cfg.hidden_size)
        self.post_attention_layernorm = Gemma2RMSNorm(cfg.hidden_size)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(cfg.hidden_size)
        self.post_feedforward_layernorm = Gemma2RMSNorm(cfg.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Sandwich Norm: Pre-norm -> Attn -> Post-norm -> Residual
        attn_out = self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), rotary_emb, attention_mask, kv_cache)
        )
        x = x + attn_out

        # Sandwich Norm: Pre-norm -> MLP -> Post-norm -> Residual
        mlp_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x + mlp_out


class Gemma2TextScaledWordEmbedding(nn.Embedding):
    """Embedding module that scales representations by sqrt(hidden_size)."""

    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(vocab_size, hidden_size, padding_idx=padding_idx)
        self.embed_scale = hidden_size ** 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class Gemma2Model(nn.Module):
    """Transformer decoder backbone for Gemma 2."""

    def __init__(self, cfg: Gemma2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Gemma2TextScaledWordEmbedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([Gemma2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = Gemma2RMSNorm(cfg.hidden_size)
        self.rotary_emb = Gemma2RotaryEmbedding(cfg)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        rotary_emb = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_emb, attention_mask, kv_cache)

        return self.norm(hidden_states)


class Gemma2ForCausalLM(nn.Module):
    """Gemma 2 language model with language modeling head and final logit softcapping."""

    def __init__(self, cfg: Gemma2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Gemma2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def tie_weights(self) -> None:
        """Ties weights between token embeddings and the LM head."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> dict:
        hidden = self.model(inputs_embeds, position_ids, attention_mask, past_key_values)
        logits = self.lm_head(hidden)
        if self.cfg.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.cfg.final_logit_softcapping) * self.cfg.final_logit_softcapping
        return {"logits": logits}