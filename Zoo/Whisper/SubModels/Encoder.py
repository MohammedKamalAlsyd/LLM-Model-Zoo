import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Whisper.SubModels.Transformer import WhisperEncoderLayer

class WhisperEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        # large-v3-turbo settings
        self.d_model = 1280
        self.num_mel_bins = 128
        self.encoder_layers = 32
        self.encoder_attention_heads = 20
        self.encoder_ffn_dim = 5120
        self.max_source_positions = 1500 # 3000 frames shrink to 1500 due to stride=2

        # 1D Convolutions shrink the time dimension from 3000 -> 1500
        self.conv1 = nn.Conv1d(self.num_mel_bins, self.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)

        # Match Hugging Face: Use an Embedding layer for positions and freeze it
        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        self.embed_positions.requires_grad_(False)

        # 32 Transformer Encoder Layers
        self.layers = nn.ModuleList([
            WhisperEncoderLayer(self.d_model, self.encoder_attention_heads, self.encoder_ffn_dim) 
            for _ in range(self.encoder_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: (Batch, 128, 3000)
        
        # 1. Convolutions & GELU
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x)) # Shape is now (Batch, 1280, 1500)

        # 2. Reshape to sequence format: (Batch, Channels, Seq) -> (Batch, Seq, Channels)
        x = x.permute(0, 2, 1) # Shape: (Batch, 1500, 1280)

        # 3. Add Positional Embeddings
        positions = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        x = x + self.embed_positions(positions)

        # 4. Pass through the 32 Transformer Layers
        for layer in self.layers:
            x = layer(x)

        # 5. Final LayerNorm
        x = self.layer_norm(x)

        return x # Output Shape: (Batch, 1500, 1280)