"""
Minimal, well-documented building blocks for a small Mistral-3 style model.

This module provides compact, readable implementations of:
 - rotary embedding helpers (Ministral3RotaryEmbedding)
 - multi-headed attention with grouped KV heads (Ministral3Attention)
 - gated MLP (Ministral3MLP)
 - RMS-style layer norm (Ministral3RMSNorm)
 - decoder layer (Ministral3DecoderLayer)
 - small causal model wrapper and LM head (Ministral3Model, Ministral3ForCausalLM)

Edits in this file are focused on clarity and correctness while preserving
public attribute names (for checkpoint compatibility) such as:
  - q_proj, k_proj, v_proj, o_proj on attention
  - embed_tokens, layers, norm, rotary_emb on the model
  - lm_head on the causal LM

Notes:
 - The model.forward returns a dictionary with "last_hidden_state" and
   "past_key_values" to match common HF-style conventions used in the code
   below (so downstream code that expects those keys works without changes).
 - We avoid renaming public attributes so that Hugging Face checkpoint weight
   loading that relies on module attribute names will continue to work.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

# Local utilities expected to exist in the repo
from Pixtral import PixtralConfig
from Zoo.Mistral3.utils.KVCache import KVCache
from Zoo.Mistral3.utils.rotate_functions import apply_rotary_pos_emb


# -----------------------
# Configuration dataclasses
# -----------------------

@dataclass
class RopeParameters:
    """Container for RoPE-related hyperparameters (kept simple here)."""
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
    """
    Minimal text-only model configuration used by the blocks in this file.
    Keep attribute names and shapes aligned with the original implementation
    for checkpoint compatibility.
    """
    attention_dropout: float = 0.0
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 14336
    max_position_embeddings: int = 262144
    num_attention_heads: int = 32
    num_hidden_layers: int = 34
    num_key_value_heads: int = 8  # For GQA-style grouped KV heads
    rms_norm_eps: float = 1e-5
    rope_parameters: dict = RopeParameters().__dict__
    vocab_size: int = 131072
    pad_token_id: Optional[int] = 11
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2


# -----------------------
# Small utility functions
# -----------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat grouped key/value heads so their count matches the attention head count.

    Input shape (expected):
        (batch, num_key_value_heads, seq_len, head_dim)

    If `n_rep == 1`, this returns the tensor unchanged. Otherwise it expands and
    reshapes to create `num_key_value_heads * n_rep` heads in place of the KV groups.

    Returns:
        Tensor of shape (batch, num_attention_heads, seq_len, head_dim)
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Insert a new axis for repeats, expand and then flatten the kv_group and repeat axis
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )  # (batch, num_kv_heads, n_rep, seq_len, head_dim)

    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _get_llama_4_attn_scale(
    positions_ids: torch.Tensor, beta: float, max_position_embeddings: int
) -> torch.Tensor:
    """
    Llama-4 style position-based attention scaling.

    Positions are bucketed in chunks of size `max_position_embeddings` and a small
    multiplicative scaling (log-based) is applied to queries from later buckets.

    Returns:
        Tensor of shape (..., 1) suitable for multiplying query vectors.
    """
    # floor(positions / max_pos_emb) yields which bucket each position lives in
    bucket_index = torch.floor(positions_ids / max_position_embeddings)
    scaling = 1.0 + beta * torch.log(1.0 + bucket_index)
    # keep last dimension for broadcasting over head_dim
    return scaling.unsqueeze(-1)


def create_causal_mask(
    config: Ministral3Config,
    inputs_embeds: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[KVCache] = None,
) -> Optional[torch.Tensor]:
    """
    Create an additive causal attention mask compatible with attention weights addition:
        attn_weights = attn_weights + attention_mask

    Returns:
        - None if inputs_embeds is None (nothing to build on)
        - Otherwise a tensor shaped (batch, 1, query_length, key_value_length) where
          allowed positions are 0.0 and masked positions are very negative (torch.finfo(dtype).min)
    """
    if inputs_embeds is None:
        return None

    batch_size, query_length, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # Determine how many past tokens are already present in the KV cache
    past_length = 0
    if past_key_values is not None:
        past_length = past_key_values.num_items()

    kv_length = past_length + query_length

    # Initialize the mask with the "very negative" value so that masked positions
    # become -inf after addition to attention logits.
    neg_inf = torch.finfo(dtype).min
    # Start with a full matrix of neg-inf and then allow lower triangle (including diagonal)
    mask = torch.full((query_length, kv_length), fill_value=neg_inf, device=device, dtype=dtype)

    # Allow attention to previous and current tokens by zeroing the lower triangular part.
    # We create an upper triangular mask and keep elements above diagonal as neg-inf.
    mask = torch.triu(mask, diagonal=1 + past_length)

    # Optional sliding-window attention constraint (if configured)
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is not None:
        # q_ids: positions of queries in absolute token space (including past)
        q_ids = torch.arange(query_length, device=device) + past_length
        kv_ids = torch.arange(kv_length, device=device)
        # distance (query_index - kv_index). Valid if <= sliding_window
        distance = q_ids[:, None] - kv_ids[None, :]
        window_allowed = distance <= sliding_window
        # Wherever the window disallows attention, set to neg-inf
        mask = torch.where(window_allowed, mask, torch.full_like(mask, neg_inf))

    # Expand to (batch, 1, query_length, kv_length)
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, query_length, kv_length)

    # Apply provided padding attention mask (shape expected to be (batch, kv_length) with 1 for keep, 0 for pad)
    if attention_mask is not None:
        # Convert to additive mask values (0 -> keep, neg_inf -> mask)
        padding_mask = (1.0 - attention_mask).to(dtype) * neg_inf  # (batch, kv_length)
        padding_mask = padding_mask[:, None, None, :]  # (batch, 1, 1, kv_length)
        mask = mask + padding_mask

    return mask


# -----------------------
# Modules
# -----------------------


class Ministral3Attention(nn.Module):
    """
    Multi-headed scaled-dot-product attention with grouped KV-head support (GQA).

    Public projection module names (q_proj, k_proj, v_proj, o_proj) are preserved
    so that checkpoints expecting these names will continue to load correctly.
    """

    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # head_dim may be explicitly present in config or computed from hidden_size / num_heads
        self.head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)

        # number of times each KV head should be repeated to match number of attention heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        # scaling for queries (classic transformer scaling)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True  # decoder-only

        # Linear projections: keep attribute names for compatibility with checkpoints
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def apply_attention(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: float = 1.0,
        dropout: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            module: module used for checking training flag for dropout (pass `self`)
            query: (batch, num_heads, q_len, head_dim)
            key: (batch, num_kv_heads, kv_len, head_dim)
            value: (batch, num_kv_heads, kv_len, head_dim)
            attention_mask: additive mask broadcastable to (batch, num_heads, q_len, kv_len)
            scaling: multiplicative scaling applied to attention logits (usually self.scaling)
            dropout: dropout probability for attention weights

        Returns:
            attn_output: (batch, q_len, num_heads, head_dim)
            attn_weights: (batch, num_heads, q_len, kv_len)
        """
        # Expand KV groups to per-attention-head tensors
        key_states = repeat_kv(key, self.num_key_value_groups)    # -> (batch, num_heads, kv_len, head_dim)
        value_states = repeat_kv(value, self.num_key_value_groups)  # -> (batch, num_heads, kv_len, head_dim)

        # compute raw attention logits: (batch, num_heads, q_len, kv_len)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in fp32 for stability and cast back to query dtype
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Dropout on attention probabilities
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        # Weighted sum -> (batch, num_heads, q_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        # Re-arrange to (batch, q_len, num_heads, head_dim) to be consistent with later reshapes
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.LongTensor,
        past_key_values: Optional[KVCache] = None,
        position_ids: Optional[torch.LongTensor] = None,  # accepted for compatibility; unused here
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute self-attention for decoder layer.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_embeddings: tuple (cos, sin) returned by the rotary embedding module
            attention_mask: additive attention mask shaped (batch, 1, q_len, kv_len)
            cache_position: (batch, seq_len) absolute position ids used for llama-4 scaling
            past_key_values: optional KVCache for incremental decoding

        Returns:
            attn_output_projected: (batch, seq_len, hidden_size)
            attn_weights: optional attention weights (batch, num_heads, q_len, kv_len)
        """
        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_len)
        # target shape for per-head projections: (batch, seq_len, num_heads, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Linear projections and reshape: -> (batch, num_heads, seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary positional embeddings to queries and keys
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Optionally scale queries based on Llama-4 style scaling factor depending on position
        llama_beta = self.config.rope_parameters.get("llama_4_scaling_beta", 0.1)
        original_max_pos = self.config.rope_parameters.get("original_max_position_embeddings", 16384)
        query_states = query_states * _get_llama_4_attn_scale(cache_position, llama_beta, original_max_pos).to(query_states.dtype)

        # If past KV cache exists, update/concatenate cached values for autoregressive decoding
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Compute attention and project back to hidden size
        attn_output, attn_weights = self.apply_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            self.attention_dropout,
        )

        # Combine per-head outputs and run final output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # (batch, seq_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Ministral3MLP(nn.Module):
    """
    Gated MLP block used in many modern transformer variants:
        output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    Public attribute names (gate_proj, up_proj, down_proj) are preserved.
    """

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Keep attribute names for checkpoint compatibility
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: up_proj(x) * SiLU(gate_proj(x)), then project down
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Ministral3RMSNorm(nn.Module):
    """
    RMS-style normalization (equivalent to T5LayerNorm behavior used by some models).

    This implementation keeps one learned scale parameter `weight` and normalizes by
    the root-mean-square (RMS) of the activations along the last dimension.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Preserve input dtype but compute variance in float32 for numeric stability
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # Scale and cast back to original dtype
        return (self.weight * x).to(input_dtype)


class Ministral3DecoderLayer(nn.Module):
    """
    Single decoder block: RMSNorm -> SelfAttention -> Residual -> RMSNorm -> MLP -> Residual.
    Public submodule names (self_attn, mlp, input_layernorm, post_attention_layernorm)
    are preserved for compatibility with checkpoints.
    """

    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Ministral3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(config)
        self.input_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Execute one decoder layer.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: additive mask (batch, 1, q_len, kv_len)
            position_ids: unused here but accepted for compatibility
            past_key_values: optional KV cache
            cache_position: (batch, seq_len) position ids used for rotary and scaling
            position_embeddings: tuple (cos, sin) from rotary embedding module

        Returns:
            Tensor of shape (batch, seq_len, hidden_size)
        """
        # Pre-norm + attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,  # accepted for compatibility
        )
        hidden_states = residual + attn_out

        # Feed-forward block with post-attention normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Ministral3RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding helper.

    Produces (cos, sin) tensors which are applied to query/key tensors using
    `apply_rotary_pos_emb`. The implementation follows the canonical RoPE
    formulation, and returns cos/sin in the same dtype as the input tensors.
    """

    inv_freq: torch.Tensor  # type annotation for register_buffer linting

    def compute_default_rope_parameters(self, config: Ministral3Config) -> Tuple[torch.Tensor, float]:
        """
        Compute the inverse frequency vector used to produce rotary angles.

        Returns:
            inv_freq: Tensor of shape (head_dim / 2,)
            attention_scaling: float multiplier applied to cos/sin (unused except kept for API consistency)
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)

        # Classic RoPE inverse frequency formula:
        # inv_freq[i] = 1 / (base ** (i / dim))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim))
        )

        attention_scaling = 1.0  # kept for API consistency
        return inv_freq, attention_scaling

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.rope_type = self.config.rope_parameters.get("rope_type", "yarn")
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config)

        # Register as non-persistent buffers so they won't be part of state_dict if not desired,
        # but will be present on the right device when moved.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cos and sin tensors for rotary positional embeddings.

        Args:
            x: tensor used to determine dtype/device (batch, seq_len, hidden_size)
            position_ids: (batch, seq_len) absolute position ids for tokens

        Returns:
            cos: (batch, seq_len, head_dim) cosine terms
            sin: (batch, seq_len, head_dim) sine terms
        """
        # inv_freq: (dim/2,)
        batch = position_ids.shape[0]

        # Expand inv_freq to (batch, dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch, -1, 1).to(x.device)

        # position_ids: (batch, seq_len) -> (batch, 1, seq_len) for matmul
        position_ids_expanded = position_ids[:, None, :].float()

        # (batch, dim/2, 1) @ (batch, 1, seq_len) -> (batch, dim/2, seq_len)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # -> (batch, seq_len, dim/2)

        # Duplicate to interleave for cos/sin pairs -> (batch, seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Ministral3Model(nn.Module):
    """
    Minimal decoder-only model composed of:
      - token embedding
      - N x decoder layers (Ministral3DecoderLayer)
      - final RMSNorm

    Public attributes are preserved: `embed_tokens`, `layers`, `norm`, `rotary_emb`.
    """

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Create decoder layers and keep them in a ModuleList with the same names as before
        self.layers = nn.ModuleList(
            [Ministral3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Ministral3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the decoder model.

        Exactly one of `input_ids` or `inputs_embeds` should be provided.

        Returns a dict consistent with typical HF outputs:
            {
                "last_hidden_state": Tensor (batch, seq_len, hidden_size),
                "past_key_values": KVCache
            }
        """
        # Ensure exactly one of input_ids / inputs_embeds is specified
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Convert ids to embeddings if needed
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Use provided KV cache or create an empty one if none was given
        if past_key_values is None:
            past_key_values = KVCache()

        # If cache_position is not provided, create a default sequence of positions
        if cache_position is None and inputs_embeds is not None:
            past_seen_tokens = past_key_values.num_items() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # If position_ids were not provided, use the cache_position (batch dim)
        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0)

        # Build causal / padding mask once and reuse for all layers
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        # Initial hidden states are token embeddings
        hidden_states = inputs_embeds

        # Compute rotary positional embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Iterate through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Return HF-style output dict for compatibility with callers in this repo
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values,
        }


class Ministral3ForCausalLM(nn.Module):
    """
    Minimal causal language modeling wrapper using Ministral3Model as backbone.

    - The lm_head uses the same weight matrix as embed_tokens for weight tying.
    - Forward returns a dict containing "logits" (batch, seq_len, vocab_size) and
      "past_key_values" (if present).
    """

    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.config = config
        self.model = Ministral3Model(config)
        self.vocab_size = config.vocab_size

        # Output projection (tied to embedding weights)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
                "logits": Tensor (batch, seq_len, vocab_size),
                "past_key_values": optional KVCache
            }
        """
        # 1) Run base transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        hidden_states = outputs["last_hidden_state"]

        # 2) Optionally slice the last N tokens for faster generation decoding
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]

        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": outputs.get("past_key_values", None),
        }
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value