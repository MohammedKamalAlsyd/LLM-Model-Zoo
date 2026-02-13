from torch import nn
from dataclasses import dataclass
from typing import Callable, Optional
import torch


@dataclass
class Gemma2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    sliding_window: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: int
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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma2Config, device=None):
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
        config: Gemma2Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        if config is None:
            config = self.config
        base = config.rope_base
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma2Attention(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = (
            config.head_dim or config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
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
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.rotary_emb = Gemma2RotaryEmbedding(config)
        
    def attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        scaling: float,
        attention_mask: Optional[torch.Tensor] = None,
        softcapping: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = repeat_kv(key_states, self.num_key_value_groups) # shape (batch, num_attention_heads, seqlen, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        
        if softcapping is not None:
            attn_weights = attn_weights / softcapping
            attn_weights = torch.tanh(attn_weights) * softcapping
            
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # upcase attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output, attn_weights
    

    def forward(
        self, 
        hidden_states,
        positional_embedding: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1 ,self.hidden_size)
        
        query_states = self.q_proj(hidden_states).view(hidden_states).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cache_kwargs = {}
        if positional_embedding is not None:
            cos, sin = positional_embedding
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )
            cache_kwargs = {'sin': sin, 'cos': cos}
            
        if past_key_value is not None:
            cache_kwargs = {**cache_kwargs, 'cache_position': cache_position}
            
        scaling = self.scaling if self.scaling is not None else self.hidden_size ** -0.5
        attn_output, attn_weights = self.attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            scaling=scaling,
            attention_mask=attention_mask,
            softcapping=self.attn_logit_softcapping,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
    
    
class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = Gemma2Attention(config, layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self, 
        hidden_states,
        positional_embedding: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        hidden_states = residual + mlp_output
        
        return hidden_states