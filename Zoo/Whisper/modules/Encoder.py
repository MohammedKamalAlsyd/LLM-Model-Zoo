"""Audio Transformer Encoder with 1D Convolutions and Sinusoidal Positional Embeddings."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Whisper.configs import WhisperEncoderConfig
from Zoo.Whisper.SubModels.Transformer import WhisperEncoderLayer


class WhisperEncoder(nn.Module):
    """Audio Encoder mapping (Batch, 128, 3000) Log-Mel spectrograms to (Batch, 1500, 1280)."""

    def __init__(self, config: Optional[WhisperEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or WhisperEncoderConfig()

        # Two 1D Convolutions downsampling the time dimension by 2x (3000 -> 1500 frames)
        self.conv1 = nn.Conv1d(self.config.num_mel_bins, self.config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.config.d_model, self.config.d_model, kernel_size=3, stride=2, padding=1)

        # Frozen Sinusoidal Positional Embeddings
        self.embed_positions = nn.Embedding(self.config.max_source_positions, self.config.d_model)
        self.embed_positions.requires_grad_(False)

        # 32 Encoder Blocks
        self.layers = nn.ModuleList([
            WhisperEncoderLayer(
                d_model=self.config.d_model,
                encoder_attention_heads=self.config.encoder_attention_heads,
                encoder_ffn_dim=self.config.encoder_ffn_dim,
            )
            for _ in range(self.config.encoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.config.d_model)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # 1. Convolutions with GELU activations
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x))  # Shape: (Batch, 1280, 1500)

        # 2. Permute to sequence format: (Batch, Sequence, Hidden)
        x = x.permute(0, 2, 1)    # Shape: (Batch, 1500, 1280)

        # 3. Add Positional Embeddings
        positions = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        x = x + self.embed_positions(positions)

        # 4. Pass through 32 Transformer Encoder Layers
        for layer in self.layers:
            x = layer(x)

        # 5. Final LayerNorm
        return self.layer_norm(x)  # Shape: (Batch, 1500, 1280)