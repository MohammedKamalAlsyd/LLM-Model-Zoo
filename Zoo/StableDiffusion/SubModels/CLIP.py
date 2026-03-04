import torch
from torch import nn
from torch.nn import functional as F

class QuickGELU(nn.Module):
    """
    Approximation of GELU activation function used by OpenAI's CLIP models.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class CLIPAttention(nn.Module):
    """
    Dedicated Attention block for CLIP. 
    Unlike the VAE or UNet, CLIP naturally uses separate linear layers for Q, K, and V.
    By structuring it this way, we match the original pretrained weights perfectly.
    """
    def __init__(self, n_heads: int, hidden_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, Dim)
        batch_size, seq_length, hidden_size = x.shape

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Heads, Head_Dim) -> (Batch_Size, Heads, Seq_Len, Head_Dim)
        q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)

        # High-performance, memory-efficient causal attention
        # is_causal=True automatically applies the upper-triangular mask needed for text generation/encoding
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # (Batch_Size, Heads, Seq_Len, Head_Dim) -> (Batch_Size, Seq_Len, Heads, Head_Dim) -> (Batch_Size, Seq_Len, Dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        # Final linear projection
        return self.out_proj(out)

class CLIPMLP(nn.Module):
    """
    Feedforward network for the CLIP encoder layer.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.activation = QuickGELU()
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class CLIPEncoderLayer(nn.Module):
    """
    A single Transformer layer (encoder) for CLIP.
    """
    def __init__(self, n_heads: int, hidden_size: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = CLIPAttention(n_heads, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture with residual connections
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class CLIPEncoder(nn.Module):
    """
    The main Transformer encoder stack.
    """
    def __init__(self, num_layers: int, n_heads: int, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(n_heads, hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class CLIPTextEmbeddings(nn.Module):
    """
    Generates combined Token and Position embeddings.
    """
    def __init__(self, n_vocab: int, hidden_size: int, max_seq_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size) 
        # This Add overhead since the position ids are the same for every batch, but it allows us to load the pretrained weights without modification.
        # The position embeddings are only 77 tokens, so it's not a huge memory cost.
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        
        # Create position IDs (0, 1, 2, ..., seq_length - 1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        token_embeddings = self.token_embedding(input_ids)
        # (Seq_Len) -> (Seq_Len, Dim)
        position_embeddings = self.position_embedding(position_ids)
        
        # Broadcasting handles the addition across the batch dimension
        return token_embeddings + position_embeddings

class CLIP(nn.Module):
    """
    Top-level CLIP Text Model.
    
    Because the internal attributes (`embeddings`, `encoder`, `final_layer_norm`)
    and sub-components strictly follow the HuggingFace / OpenCLIP naming conventions, 
    you can directly load standard weights without a conversion script.
    """
    def __init__(self):
        super().__init__()
        # Standard configuration for Stable Diffusion v1.5 text encoder
        vocab_size = 49408
        hidden_size = 768
        max_seq_length = 77
        num_layers = 12
        n_heads = 12

        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_seq_length)
        self.encoder = CLIPEncoder(num_layers, n_heads, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Enforce long data type for embedding lookup
        input_ids = input_ids.type(torch.long)
        
        # 1. Embeddings
        x = self.embeddings(input_ids)

        # 2. Transformer Encoder Layers
        x = self.encoder(x)

        # 3. Final Layer Normalization
        output = self.final_layer_norm(x)
        
        return output