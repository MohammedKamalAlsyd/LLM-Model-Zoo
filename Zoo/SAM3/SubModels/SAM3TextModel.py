import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # attention_mask is already a combined 4D [B, 1, S, S] float causal mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, s, self.embed_dim)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096, num_heads: int = 16):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096, num_heads: int = 16, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, intermediate_size, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class CLIPTextEmbeddings(nn.Module):
    position_ids: torch.Tensor
    def __init__(self, vocab_size: int = 49408, hidden_size: int = 1024, max_position_embeddings: int = 32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(self.position_ids[:, :seq_length])
        return inputs_embeds + position_embeddings


class CLIPTextTransformer(nn.Module):
    """Matches exact checkpoint hierarchy `text_encoder.text_model.*`."""
    def __init__(self, vocab_size: int = 49408, hidden_size: int = 1024, max_positions: int = 32):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_positions)
        self.encoder = CLIPEncoder(hidden_size=hidden_size, intermediate_size=4096, num_heads=16, num_layers=24)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        b, seq_len = input_ids.shape
        device = input_ids.device
        dtype = hidden_states.dtype

        # 1. Base Causal Mask: upper triangular filled with -inf [1, 1, seq_len, seq_len]
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]

        # 2. Combine with padding mask (converting int64 [B, seq_len] to additive float [B, 1, 1, seq_len])
        if attention_mask is not None:
            pad_mask = torch.where(attention_mask[:, None, None, :].bool(), 0.0, float("-inf")).to(dtype)
            combined_mask = causal_mask + pad_mask
        else:
            combined_mask = causal_mask

        hidden_states = self.encoder(hidden_states, attention_mask=combined_mask)
        return self.final_layer_norm(hidden_states)


class Sam3TextEncoder(nn.Module):
    """
    Standard CLIPTextModelWithProjection wrapper.
    Matches checkpoint keys:
      - text_encoder.text_model.*
      - text_encoder.text_projection.weight
    """
    def __init__(self, hidden_size: int = 1024, projection_dim: int = 512):
        super().__init__()
        self.text_model = CLIPTextTransformer(hidden_size=hidden_size)
        self.text_projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.text_model(input_ids, attention_mask=attention_mask)