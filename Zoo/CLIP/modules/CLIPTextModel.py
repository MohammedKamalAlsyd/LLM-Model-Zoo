"""CLIP Text Transformer Tower matching Hugging Face weights and Stable Diffusion text encoders."""

import torch
import torch.nn as nn
from Zoo.CLIP.configs import CLIPTextConfig
from Zoo.CLIP.modules.CLIPEncoder import CLIPEncoder


class CLIPTextEmbeddings(nn.Module):
    """Combines discrete token IDs and absolute positional indices."""
    position_ids: torch.Tensor
    
    def __init__(self, cfg: CLIPTextConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(cfg.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args:

            input_ids: LongTensor of shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size).
        """
        seq_length = input_ids.shape[-1]
        inputs_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(self.position_ids[:, :seq_length])
        return inputs_embeds + pos_embeds


class CLIPTextTransformer(nn.Module):
    """Core Text Transformer preserving exact `text_model.*` checkpoint hierarchy."""

    def __init__(self, cfg: CLIPTextConfig) -> None:
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(cfg)
        self.encoder = CLIPEncoder(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads,
            num_layers=cfg.num_hidden_layers,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_pooled: bool = True,
    ) -> torch.Tensor:
        """Args:

            input_ids: LongTensor of shape (batch_size, seq_len).
            return_pooled: If True, returns EOS token vector (B, D). If False,
                           returns sequence states (B, S, D) for UNet cross-attention.

        Returns:
            Tensor of shape (batch_size, hidden_size) or (batch_size, seq_len, hidden_size).
        """
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, causal_attention_mask=True)
        last_hidden_state = self.final_layer_norm(hidden_states)

        if not return_pooled:
            return last_hidden_state

        # Find EOS token position via argmax to extract sequence summary vector
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.argmax(dim=-1),
        ]
        return pooled_output


class CLIPTextModel(nn.Module):
    """Top-level Text Model matching standard Hugging Face checkpoints."""

    def __init__(self, cfg: CLIPTextConfig) -> None:
        super().__init__()
        self.text_model = CLIPTextTransformer(cfg)

    def forward(self, input_ids: torch.Tensor, return_pooled: bool = True) -> torch.Tensor:
        return self.text_model(input_ids, return_pooled=return_pooled)