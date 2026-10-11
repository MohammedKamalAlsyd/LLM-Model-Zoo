"""High-Performance Transformer Attention and Layers with Stateful Key-Value Caching."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperAttention(nn.Module):
    """Multi-Head Attention supporting both Self-Attention and Cross-Attention with KV-Caching."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # In Whisper: Q, V, Out have bias=True; K has bias=False
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Runs Multi-Head Attention using PyTorch FlashAttention (SDPA) and KV caches.

        Args:
            hidden_states: Tensor of shape (Batch, Query_Len, Embed_Dim).
            key_value_states: Optional audio encoder states for Cross-Attention (Batch, Audio_Len, Embed_Dim).
            past_key_value: Tuple of cached (Key, Value) tensors from prior generation steps.
            use_cache: Boolean indicating whether to return updated KV caches.
            is_causal: Boolean indicating whether to apply causal triangular masking.

        Returns:
            attn_output: Attention output tensor of shape (Batch, Query_Len, Embed_Dim).
            present_key_value: Updated (Key, Value) tuple if use_cache is True, else None.
        """
        batch_size, q_len, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        # 1. Project Query
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Project Key & Value (Handling Cross-Attention and Self-Attention caches)
        if is_cross_attention:
            # Cross-Attention: Audio keys and values are static across all token generation steps
            if past_key_value is not None:
                k, v = past_key_value
            else:
                k = self.k_proj(key_value_states)
                v = self.v_proj(key_value_states)
                k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # Self-Attention: Keys and values accumulate token by token
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            k = k.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # 3. Scaled Dot-Product Attention with FlashAttention
        # When generating token-by-token with KV-cache (q_len == 1), causal masking is not needed
        apply_causal = is_causal and (q_len > 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=apply_causal)

        # 4. Reshape back to (Batch, Query_Len, Embed_Dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)
        return self.out_proj(attn_output), present_key_value


class WhisperEncoderLayer(nn.Module):
    """Single Audio Transformer block utilizing Pre-LayerNorm."""

    def __init__(self, d_model: int, encoder_attention_heads: int, encoder_ffn_dim: int) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = WhisperAttention(d_model, encoder_attention_heads)

        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        residual = hidden_states
        h = self.self_attn_layer_norm(hidden_states)
        h, _ = self.self_attn(h, is_causal=False)
        hidden_states = residual + h

        # Pre-LN Feed-Forward (MLP)
        residual = hidden_states
        h = self.final_layer_norm(hidden_states)
        h = self.fc2(F.gelu(self.fc1(h)))
        return residual + h


class WhisperDecoderLayer(nn.Module):
    """Text Decoder block with Masked Self-Attention, Audio Cross-Attention, and KV Caches."""

    def __init__(self, d_model: int, decoder_attention_heads: int, decoder_ffn_dim: int) -> None:
        super().__init__()
        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = WhisperAttention(d_model, decoder_attention_heads)

        # Cross-Attention to audio features
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = WhisperAttention(d_model, decoder_attention_heads)

        # MLP
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. Causal Self-Attention
        residual = hidden_states
        h = self.self_attn_layer_norm(hidden_states)
        h, present_self_kv = self.self_attn(
            hidden_states=h,
            past_key_value=past_self_kv,
            use_cache=use_cache,
            is_causal=True,
        )
        hidden_states = residual + h

        # 2. Audio Cross-Attention
        residual = hidden_states
        h = self.encoder_attn_layer_norm(hidden_states)
        h, present_cross_kv = self.encoder_attn(
            hidden_states=h,
            key_value_states=encoder_hidden_states,
            past_key_value=past_cross_kv,
            use_cache=use_cache,
            is_causal=False,
        )
        hidden_states = residual + h

        # 3. Feed-Forward
        residual = hidden_states
        h = self.final_layer_norm(hidden_states)
        h = self.fc2(F.gelu(self.fc1(h)))
        hidden_states = residual + h

        return hidden_states, present_self_kv, present_cross_kv