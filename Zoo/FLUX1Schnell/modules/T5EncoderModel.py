"""T5-v1.1-XXL text encoder implementation for contextual prompt representation in FLUX."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.FLUX1Schnell.configs import T5Config


@dataclass
class T5EncoderOutput:
    """Output container for the T5 encoder stack."""
    last_hidden_state: torch.Tensor


class T5LayerNorm(nn.Module):
    """RMSNorm without bias and mean subtraction."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)


class T5DenseGatedActDense(nn.Module):
    """Gated GELU Feed-Forward Network."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_gelu = F.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        return self.wo(hidden_gelu * hidden_linear)


class T5LayerFF(nn.Module):
    """Feed-Forward layer with pre-LayerNorm and residual connection."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Attention(nn.Module):
    """Multi-Head Self-Attention with learned relative position bias."""

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        assert self.has_relative_attention_bias, "relative_attention_bias is not defined on this layer"
        values = self.relative_attention_bias(relative_position_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = x.shape

        q = self.q(x).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # Standard T5 Attention does not divide scores by sqrt(d_k)
        scores = torch.matmul(q, k.transpose(2, 3))

        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(seq_length, seq_length, device=x.device)

        if position_bias is not None:
            scores = scores + position_bias

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_dim)
        return self.o(attn_output), position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normed = self.layer_norm(x)
        out, position_bias = self.SelfAttention(normed, mask=mask, position_bias=position_bias)
        return x + out, position_bias


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, position_bias = self.layer[0](x, mask=mask, position_bias=position_bias)
        x = self.layer[1](x)
        return x, position_bias


class T5Stack(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> T5EncoderOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask.to(inputs_embeds.dtype)) * -1e9
            extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)

        hidden_states = inputs_embeds
        position_bias: Optional[torch.Tensor] = None

        for layer_module in self.block:
            hidden_states, position_bias = layer_module(
                hidden_states, mask=extended_attention_mask, position_bias=position_bias
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return T5EncoderOutput(last_hidden_state=hidden_states)


class T5EncoderModel(nn.Module):
    """Standalone T5-XXL Encoder module for prompt sequence representation."""

    def __init__(self, config: Optional[T5Config] = None) -> None:
        super().__init__()
        self.config = config or T5Config()
        self.shared = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.encoder = T5Stack(self.config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> T5EncoderOutput:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

    def tie_weights(self) -> None:
        """Ties shared token embedding weights with encoder embedding weights."""
        self.encoder.embed_tokens.weight = self.shared.weight