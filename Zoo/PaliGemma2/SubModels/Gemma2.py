"""Gemma2 Language Model Components.

This module provides a complete implementation of the Gemma2 language model,
including configuration, embeddings, attention mechanisms, and transformer layers.
Gemma2 is an efficient, production-ready large language model based on the
transformer architecture with rotary position embeddings and grouped query attention.

Key Features:
    - Grouped Query Attention (GQA) for efficient inference
    - Rotary Position Embeddings (RoPE) for better length extrapolation
    - RMSNorm for stable training
    - Soft capping on attention logits for numerical stability
    - GeLU activation with tanh approximation

Classes:
    Gemma2Config: Configuration dataclass for model hyperparameters
    Gemma2RMSNorm: Root Mean Square Layer Normalization
    Gemma2MLP: Feed-forward network with gating
    Gemma2RotaryEmbedding: Rotary positional embeddings
    Gemma2Attention: Multi-head grouped query attention
    Gemma2DecoderLayer: Single transformer decoder block
    Gemma2Model: Complete language model
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "Gemma2Config",
    "Gemma2RMSNorm",
    "Gemma2MLP",
    "Gemma2RotaryEmbedding",
    "Gemma2Attention",
    "Gemma2DecoderLayer",
]


@dataclass
class Gemma2Config:
    """Configuration class for Gemma2 model hyperparameters.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Hidden dimension of the model.
        intermediate_size (int): Intermediate dimension in MLP layers.
        num_attention_heads (int): Number of attention heads.
        num_hidden_layers (int): Number of transformer layers.
        num_key_value_heads (int): Number of key-value heads for GQA.
        sliding_window (int): Window size for sliding window attention.
        head_dim (int): Dimension per attention head.
        max_position_embeddings (int): Maximum sequence length.
        rms_norm_eps (float): Epsilon for RMSNorm stability.
        pad_token_id (int): Padding token ID.
        eos_token_id (int): End-of-sequence token ID.
        bos_token_id (int): Beginning-of-sequence token ID.
        image_token_id (int): Special token ID for images.
        attention_dropout (float): Dropout rate in attention.
        rope_base (int): Base for RoPE embeddings.
        query_pre_attn_scalar (float): Scaling factor for queries.
        attn_logit_softcapping (Optional[float]): Softcap for attention logits.
        final_logit_softcapping (Optional[float]): Softcap for final logits.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    sliding_window: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    image_token_id: int
    attention_dropout: float
    rope_base: int
    query_pre_attn_scalar: float
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None


class Gemma2RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Implements RMSNorm as described in "Root Mean Square Layer Normalization"

    Attributes:
        weight (nn.Parameter): Learnable gain parameter.
        eps (float): Small value for numerical stability.

    Shape:
        - Input: `(*, hidden_size)`
        - Output: `(*, hidden_size)`
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm layer.

        Args:
            dim (int): Dimension of the input features.
            eps (float): Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization.

        Args:
            x (torch.Tensor): Input tensor of shape `(*, dim)`.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor scaled by learned weight.
        """
        output = self._norm(x.float())
        # Gemma2 applies scaling as (x_norm * weight).to(dtype) rather than
        # (x * weight).to(dtype) for numerical accuracy (see issue #29402)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma2MLP(nn.Module):
    """Feed-Forward Network with Gating (SwiGLU).

    Implements a gated linear unit (GLU) variant with GELU activation.
    The architecture is: gate_proj * activation(up_proj) -> down_proj

    This design allows the model to learn non-linear transformations with
    gating, enabling more expressive representations.

    Attributes:
        hidden_size (int): Input/output dimension.
        intermediate_size (int): Hidden dimension (typically 2-4x hidden_size).
        gate_proj (nn.Linear): Gate projection layer.
        up_proj (nn.Linear): Up projection layer.
        down_proj (nn.Linear): Down projection layer.
        act_fn (nn.GELU): GELU activation with tanh approximation.

    Shape:
        - Input: `(batch_size, seq_len, hidden_size)`
        - Output: `(batch_size, seq_len, hidden_size)`

    Example:
        >>> mlp = Gemma2MLP(config)
        >>> x = torch.randn(2, 10, 2048)  # (batch, seq_len, hidden_size)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([2, 10, 2048])
    """

    def __init__(self, config: Gemma2Config) -> None:
        """Initialize MLP layer.

        Args:
            config (Gemma2Config): Model configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward transformation.

        Shape Trace:
            x: (batch, seq_len, hidden_size)
            gate_proj(x): (batch, seq_len, intermediate_size)
            up_proj(x): (batch, seq_len, intermediate_size)
            gate * activation(up): (batch, seq_len, intermediate_size)
            down_proj(...): (batch, seq_len, hidden_size)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma2RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Implements RoPE as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864). This approach rotates query and key vectors in the
    2D subspaces of the embedding space, enabling better length extrapolation.

    Attributes:
        inv_freq (torch.Tensor): Inverse frequencies for position computations.
        original_inv_freq (torch.Tensor): Original inverse frequencies (for reference).
        attention_scaling (float): Scaling factor for attention computations.

    Shape:
        - Input x: `(batch_size, seq_len, hidden_size)`
        - Input position_ids: `(batch_size, seq_len)`
        - Output cos: `(batch_size, seq_len, head_dim)`
        - Output sin: `(batch_size, seq_len, head_dim)`
    """

    inv_freq: torch.Tensor  # For type checking during registration

    def __init__(self, config: Gemma2Config, device: Optional[torch.device] = None) -> None:
        """Initialize rotary embeddings.

        Args:
            config (Gemma2Config): Model configuration.
            device (Optional[torch.device]): Device to place tensors on.
        """
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def compute_default_rope_parameters(
        self,
        config: Gemma2Config,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Compute inverse frequencies for RoPE.

        Args:
            config (Gemma2Config): Model configuration.
            device (Optional[torch.device]): Device for tensor placement.
            seq_len (Optional[int]): Unused, kept for API compatibility.

        Returns:
            Tuple[torch.Tensor, float]: Inverse frequencies and attention scaling factor.
        """
        base = config.rope_base
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in standard RoPE

        # Compute inverse frequencies: 1 / (base^(2i/d))
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float32
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin embeddings for rotary position encoding.

        Shape Trace:
            inv_freq_expanded: (batch, head_dim//2, 1)
            position_ids_expanded: (batch, 1, seq_len)
            freqs (matmul): (batch, head_dim//2, seq_len) -> (batch, seq_len, head_dim//2)
            emb (concat): (batch, seq_len, head_dim)
            cos, sin: (batch, seq_len, head_dim) * attention_scaling

        Args:
            x (torch.Tensor): Input tensor for device reference.
            position_ids (torch.Tensor): Token position IDs of shape (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine embeddings,
                each of shape (batch_size, seq_len, head_dim).
        """
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input.

    This is used in applying rotary position embeddings. It creates
    the rotation by applying [-x2, x1] where x = [x1, x2].

    Args:
        x (torch.Tensor): Input tensor of shape `(*, head_dim)`.

    Returns:
        torch.Tensor: Rotated tensor of same shape as input.

    Example:
        >>> x = torch.tensor([[1., 2., 3., 4.]])
        >>> rotate_half(x)
        tensor([[-2., 1., -4., 3.]])
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Rotates the query and key vectors in 2D subspaces using cos/sin embeddings.
    This enables better generalization to longer sequences than absolute positions.

    Shape Trace:
        q: (batch, seq_len, num_heads, head_dim)
        k: (batch, seq_len, num_heads, head_dim)
        cos: (batch, seq_len, head_dim) -> (1, 1, head_dim, 1) after unsqueeze
        sin: (batch, seq_len, head_dim) -> (1, 1, head_dim, 1) after unsqueeze
        q_embed: (batch, seq_len, num_heads, head_dim)
        k_embed: (batch, seq_len, num_heads, head_dim)

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        cos (torch.Tensor): Cosine part of position embeddings.
        sin (torch.Tensor): Sine part of position embeddings.
        unsqueeze_dim (int): Dimension along which to unsqueeze cos/sin
            for broadcasting. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.

    Note:
        The unsqueeze_dim parameter controls broadcasting:
        - unsqueeze_dim=1: cos/sin shape (batch, 1, seq_len, head_dim)
        - unsqueeze_dim=2: cos/sin shape (batch, seq_len, 1, head_dim)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key-value heads to match number of query heads.

    For Grouped Query Attention (GQA), key and value tensors have fewer heads
    than query tensors. This function repeats them to match.

    Shape Trace:
        Input: (batch, num_kv_heads, seq_len, head_dim)
        After expand: (batch, num_kv_heads, n_rep, seq_len, head_dim)
        Output: (batch, num_kv_heads * n_rep, seq_len, head_dim)

    Args:
        hidden_states (torch.Tensor): Key or value tensor of shape
            (batch, num_kv_heads, seq_len, head_dim).
        n_rep (int): Repetition factor (num_query_heads / num_kv_heads).

    Returns:
        torch.Tensor: Repeated tensor of shape (batch, num_query_heads, seq_len, head_dim).

    Example:
        >>> x = torch.randn(2, 4, 10, 64)  # (batch=2, kv_heads=4, seq_len=10, head_dim=64)
        >>> y = repeat_kv(x, n_rep=2)  # Repeat 2x for 8 query heads
        >>> y.shape
        torch.Size([2, 8, 10, 64])
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma2Attention(nn.Module):
    """Multi-head Grouped Query Attention (GQA) module.

    Implements scaled dot-product attention with Grouped Query Attention,
    which uses fewer key-value heads than query heads for efficiency while
    maintaining model capacity. Includes soft capping for numerical stability.

    Attributes:
        hidden_size (int): Model hidden dimension.
        num_attention_heads (int): Number of query attention heads.
        num_key_value_heads (int): Number of key-value heads.
        head_dim (int): Dimension per attention head.
        scaling (float): Attention weight scaling factor.
        attn_logit_softcapping (Optional[float]): Softcap threshold for attention logits.

    Shape:
        - Input hidden_states: `(batch, seq_len, hidden_size)`
        - Output: `(batch, seq_len, hidden_size)`

    Example:
        >>> attention = Gemma2Attention(config, layer_idx=0)
        >>> hidden = torch.randn(2, 10, 2048)
        >>> attn_out, attn_weights = attention(hidden)
        >>> attn_out.shape
        torch.Size([2, 10, 2048])
    """

    def __init__(self, config: Gemma2Config, layer_idx: int) -> None:
        """Initialize attention layer.

        Args:
            config (Gemma2Config): Model configuration.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout: float = config.attention_dropout
        self.hidden_size: int = config.hidden_size
        self.num_attention_heads: int = config.num_attention_heads
        self.head_dim: int = (
            config.head_dim or config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads: int = config.num_key_value_heads
        self.num_key_value_groups: int = self.num_attention_heads // self.num_key_value_heads
        self.scaling: float = config.query_pre_attn_scalar**-0.5
        self.attention_dropout_layer = nn.Dropout(config.attention_dropout)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.attn_logit_softcapping: Optional[float] = config.attn_logit_softcapping
        self.rotary_emb = Gemma2RotaryEmbedding(config)

    def attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        scaling: float,
        attention_mask: Optional[torch.Tensor] = None,
        softcapping: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention with optional soft capping.

        Shape Trace:
            query_states: (batch, num_heads, seq_len, head_dim)
            key_states: (batch, num_kv_heads, seq_len, head_dim)
            key_states (after repeat): (batch, num_heads, seq_len, head_dim)
            attn_weights (after matmul): (batch, num_heads, seq_len, seq_len)
            attn_output: (batch, num_heads, seq_len, head_dim)

        Args:
            query_states (torch.Tensor): Query tensor.
            key_states (torch.Tensor): Key tensor (may have fewer heads).
            value_states (torch.Tensor): Value tensor (may have fewer heads).
            scaling (float): Attention scaling factor.
            attention_mask (Optional[torch.Tensor]): Causal or sliding window mask.
            softcapping (Optional[float]): Softcap threshold for attention logits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and attention weights.
        """
        # Repeat key/value heads to match query heads (GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        # Apply soft capping if specified
        if softcapping is not None:
            attn_weights = attn_weights / softcapping
            attn_weights = torch.tanh(attn_weights) * softcapping

        # Apply attention mask (causal mask, sliding window, padding)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in float32 for numerical stability
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply grouped query attention.

        Shape Trace:
            hidden_states: (batch, seq_len, hidden_size)
            query_states: (batch, seq_len, hidden_size) -> proj -> (batch, seq_len, num_heads*head_dim)
                         -> reshape -> (batch, num_heads, seq_len, head_dim)
            key_states: (batch, seq_len, num_kv_heads*head_dim) -> (batch, num_kv_heads, seq_len, head_dim)
            After RoPE: positions encoded in q and k
            attn_output: (batch, num_heads, seq_len, head_dim)
            output: (batch, seq_len, hidden_size)

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            positional_embedding (Optional[Tuple]): (cos, sin) rotary embeddings.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            past_key_value (Optional[Tuple]): Cached key-value for incremental decoding.
            cache_position (Optional[torch.LongTensor]): Positions in cache.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Attention output and weights.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.hidden_size)

        # Project to Q, K, V and reshape for multi-head attention
        query_states = (
            self.q_proj(hidden_states)
            .view(*input_shape, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(*input_shape, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(*input_shape, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cache_kwargs = {}
        if positional_embedding is not None:
            cos, sin = positional_embedding
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )
            cache_kwargs = {"sin": sin, "cos": cos}

        if past_key_value is not None:
            cache_kwargs = {**cache_kwargs, "cache_position": cache_position}

        scaling = self.scaling if self.scaling is not None else self.hidden_size**-0.5
        attn_output, attn_weights = self.attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            scaling=scaling,
            attention_mask=attention_mask,
            softcapping=self.attn_logit_softcapping,
        )

        # Merge heads and project to output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
    
    
class Gemma2DecoderLayer(nn.Module):
    """Single transformer decoder layer with attention and feed-forward.

    Combines multi-head grouped query self-attention with a gated feed-forward
    network (MLP) in a residual configuration. Uses pre-normalization architecture
    (normalize before applying each sub-layer).

    The layer structure:
        1. LayerNorm(hidden_states)
        2. Self-Attention + Residual
        3. LayerNorm(after_attention)
        4. MLP + Residual

    Attributes:
        self_attn (Gemma2Attention): Grouped query attention module.
        mlp (Gemma2MLP): Feed-forward network module.
        input_layernorm (Gemma2RMSNorm): Normalization before attention.
        post_attention_layernorm (Gemma2RMSNorm): Normalization after attention.
        pre_feedforward_layernorm (Gemma2RMSNorm): Normalization before MLP.
        post_feedforward_layernorm (Gemma2RMSNorm): Normalization after MLP.

    Shape:
        - Input hidden_states: `(batch, seq_len, hidden_size)`
        - Output: `(batch, seq_len, hidden_size)`

    Example:
        >>> layer = Gemma2DecoderLayer(config, layer_idx=0)
        >>> hidden = torch.randn(2, 10, 2048)
        >>> output = layer(hidden)
        >>> output.shape
        torch.Size([2, 10, 2048])
    """

    def __init__(self, config: Gemma2Config, layer_idx: int) -> None:
        """Initialize decoder layer.

        Args:
            config (Gemma2Config): Model configuration.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.hidden_size: int = config.hidden_size
        self.layer_idx: int = layer_idx

        self.self_attn = Gemma2Attention(config, layer_idx)
        self.mlp = Gemma2MLP(config)

        # Pre-normalization layers
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Apply decoder layer with attention and feed-forward.

        Shape Trace:
            Input hidden_states: (batch, seq_len, hidden_size)
            After attention:
              - norm: (batch, seq_len, hidden_size)
              - attn_out: (batch, seq_len, hidden_size)
              - residual add: (batch, seq_len, hidden_size)
            After MLP:
              - norm: (batch, seq_len, hidden_size)
              - mlp_out: (batch, seq_len, hidden_size)
              - residual add: (batch, seq_len, hidden_size)
            Output: (batch, seq_len, hidden_size)

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            positional_embedding (Optional[Tuple]): (cos, sin) rotary embeddings.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            past_key_value (Optional[Tuple]): Cached KV for incremental decoding.
            cache_position (Optional[torch.LongTensor]): Cache positions.

        Returns:
            torch.Tensor: Output hidden states of same shape as input.
        """
        # Self-attention block with pre-normalization and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            positional_embedding=positional_embedding,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + attn_output

        # Feed-forward block with pre-normalization and residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states