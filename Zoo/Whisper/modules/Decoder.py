"""Text Transformer Decoder with Token Embeddings and Stateful KV-Cache Management."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from Zoo.Whisper.configs import WhisperDecoderConfig
from Zoo.Whisper.SubModels.Transformer import WhisperDecoderLayer


class WhisperDecoder(nn.Module):
    """Text Decoder predicting transcription tokens autoregressively."""

    def __init__(self, config: Optional[WhisperDecoderConfig] = None) -> None:
        super().__init__()
        self.config = config or WhisperDecoderConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.embed_positions = nn.Embedding(self.config.max_target_positions, self.config.d_model)

        # 4 Layers for large-v3-turbo (32 for standard large-v3)
        self.layers = nn.ModuleList([
            WhisperDecoderLayer(
                d_model=self.config.d_model,
                decoder_attention_heads=self.config.decoder_attention_heads,
                decoder_ffn_dim=self.config.decoder_ffn_dim,
            )
            for _ in range(self.config.decoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]]]:
        """Runs the decoder with automatic position slicing for KV caching.

        Args:
            input_ids: Token IDs of shape (Batch, Seq_Len). When caching, Seq_Len == 1.
            encoder_hidden_states: Audio representations of shape (Batch, 1500, 1280).
            past_key_values: Nested list of (self_kv, cross_kv) per layer from previous steps.
            use_cache: Whether to return updated KV caches.

        Returns:
            hidden_states: Output representations of shape (Batch, Seq_Len, 1280).
            next_key_values: Updated KV cache for all layers if use_cache is True.
        """
        seq_len = input_ids.shape[1]

        # Calculate position index (handles token-by-token generation with KV caching)
        if past_key_values is not None:
            past_length = past_key_values[0][0][0].shape[2]
            positions = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device)
        else:
            positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)

        # Embed Tokens and add Learned Positional Embeddings
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)

        next_key_values = [] if use_cache else None

        # Pass through Decoder Layers
        for i, layer in enumerate(self.layers):
            past_self = past_key_values[i][0] if past_key_values is not None else None
            past_cross = past_key_values[i][1] if past_key_values is not None else None

            x, present_self, present_cross = layer(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                past_self_kv=past_self,
                past_cross_kv=past_cross,
                use_cache=use_cache,
            )

            if use_cache:
                next_key_values.append((present_self, present_cross))

        x = self.layer_norm(x)
        return x, next_key_values