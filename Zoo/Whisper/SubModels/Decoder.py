import torch
import torch.nn as nn
from Zoo.Whisper.SubModels.Transformer import WhisperDecoderLayer

class WhisperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # large-v3-turbo settings
        self.d_model = 1280
        self.vocab_size = 51866 # Note: v3 vocab is slightly larger than v1/v2
        
        # *** TURBO SPEEDUP ***
        # Standard large-v3 uses 32 layers. Turbo only uses 4!
        self.decoder_layers = 4 
        
        self.decoder_attention_heads = 20
        self.decoder_ffn_dim = 5120
        self.max_target_positions = 448 # Maximum output tokens

        # Token Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)
        
        # Learned Positional Embeddings (Unlike the Encoder's fixed sinusoids)
        self.embed_positions = nn.Embedding(self.max_target_positions, self.d_model)

        # 4 Transformer Decoder Layers
        self.layers = nn.ModuleList([
            WhisperDecoderLayer(self.d_model, self.decoder_attention_heads, self.decoder_ffn_dim) 
            for _ in range(self.decoder_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # input_ids: (Batch, Seq_Len)
        # encoder_hidden_states: (Batch, 1500, 1280)
        
        seq_len = input_ids.shape[1]
        
        # 1. Embed Tokens
        x = self.embed_tokens(input_ids) # (Batch, Seq_Len, 1280)
        
        # 2. Add Learned Positional Embeddings
        # We generate a sequence of position IDs: [0, 1, 2, ..., seq_len - 1]
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        x = x + self.embed_positions(positions)

        # 3. Pass through the 4 Transformer Layers (Auto-Regressive + Cross-Attention)
        for layer in self.layers:
            x = layer(x, encoder_hidden_states)

        # 4. Final LayerNorm
        x = self.layer_norm(x)

        return x # Output Shape: (Batch, Seq_Len, 1280)