from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class WhisperAttention(nn.Module):
    """
    Multi-head attention used for both Self-Attention and Cross-Attention.
    Uses PyTorch's highly optimized Flash Attention (SDPA).
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Whisper includes biases on Q and V, but NOT on K.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        key_value_states: Optional[torch.Tensor] = None, 
        is_causal: bool = False
    ) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # If key_value_states is provided, this is Cross-Attention.
        # If not, it is Self-Attention, so we use hidden_states for K and V.
        if key_value_states is None:
            key_value_states = hidden_states

        kv_seq_len = key_value_states.shape[1]

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(key_value_states)
        v = self.v_proj(key_value_states)

        # Reshape for multi-head attention: (Batch, Seq_Len, Heads, Head_Dim) -> (Batch, Heads, Seq_Len, Head_Dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # High-performance PyTorch Scaled Dot-Product Attention
        # is_causal=True automatically applies the triangle mask needed for the Decoder
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape back to (Batch, Seq_Len, Embed_Dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    """
    A single Transformer block for the Audio Encoder.
    Uses Pre-LayerNorm architecture.
    """
    def __init__(self, d_model: int, encoder_attention_heads: int, encoder_ffn_dim: int):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = WhisperAttention(d_model, encoder_attention_heads)
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Pre-LN Self-Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, is_causal=False)
        hidden_states = residual + hidden_states

        # 2. Pre-LN Feed Forward (MLP)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperDecoderLayer(nn.Module):
    """
    A single Transformer block for the Text Decoder.
    Contains Self-Attention (Masked), Cross-Attention (Audio-Guided), and an MLP.
    """
    def __init__(self, d_model: int, decoder_attention_heads: int, decoder_ffn_dim: int):
        super().__init__()
        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = WhisperAttention(d_model, decoder_attention_heads)
        
        # Cross-Attention
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = WhisperAttention(d_model, decoder_attention_heads)
        
        # MLP
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, d_model)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Causal Self-Attention (Look at previous text tokens)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # is_causal=True prevents the model from looking into the future!
        hidden_states = self.self_attn(hidden_states, is_causal=True)
        hidden_states = residual + hidden_states

        # 2. Cross-Attention (Look at the audio features)
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        # key_value_states come from the audio encoder
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states, 
            key_value_states=encoder_hidden_states, 
            is_causal=False
        )
        hidden_states = residual + hidden_states

        # 3. Feed Forward (MLP)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states