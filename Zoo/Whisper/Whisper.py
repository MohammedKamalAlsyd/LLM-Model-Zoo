"""Unified Whisper Architecture for Conditional Speech Generation and Fast KV-Cached Decoding."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Whisper.configs import WhisperConfig
from Zoo.Whisper.modules.Decoder import WhisperDecoder
from Zoo.Whisper.modules.Encoder import WhisperEncoder


class TiedLinear:
    """Functional linear projection sharing weights with an Embedding without creating a duplicate state_dict parameter."""

    def __init__(self, embedding: nn.Embedding) -> None:
        self._embedding = embedding

    @property
    def weight(self) -> torch.Tensor:
        return self._embedding.weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._embedding.weight)


class WhisperModel(nn.Module):
    """Encoder-Decoder Backbone matching checkpoint hierarchy `model.*`."""

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config.encoder_config)
        self.decoder = WhisperDecoder(config.decoder_config)

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        encoder_hidden_states = self.encoder(input_features)
        decoder_outputs, next_kv = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return decoder_outputs, next_kv


class WhisperForConditionalGeneration(nn.Module):
    """Top-level Whisper model matching official checkpoints with tied LM Head."""

    def __init__(self, config: Optional[WhisperConfig] = None) -> None:
        super().__init__()
        self.config = config or WhisperConfig()
        self.model = WhisperModel(self.config)

        # LM Head tied directly to the decoder embedding weights
        # Uses TiedLinear so 'proj_out.weight' is not expected as an independent key in state_dict
        self.proj_out = TiedLinear(self.model.decoder.embed_tokens)

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        hidden_states, next_kv = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.proj_out(hidden_states)
        return {"logits": logits, "past_key_values": next_kv}

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        tokenizer: Any,
        language: str = "en",
        task: str = "transcribe",
        without_timestamps: bool = True,
        max_new_tokens: int = 448,
    ) -> List[int]:
        """Runs fast autoregressive generation utilizing KV-Caching ($O(N)$ execution).

        Args:
            input_features: Log-Mel spectrogram tensor of shape (1, 128, 3000).
            tokenizer: WhisperTokenizer instance.
            language: Target ISO language code (e.g., 'en', 'es', 'fr', 'de', 'ar', 'zh').
            task: Either 'transcribe' or 'translate'.
            without_timestamps: Whether to suppress timestamp tokens.
            max_new_tokens: Maximum tokens to generate per 30s window.

        Returns:
            generated_ids: List of generated token IDs (excluding initial prompt tokens).
        """
        device = input_features.device

        # 1. Run Audio Encoder ONCE
        encoder_hidden_states = self.model.encoder(input_features)

        # 2. Formulate Prompt Tokens
        prompt_tokens = ["<|startoftranscript|>"]
        lang_token = f"<|{language}|>" if f"<|{language}|>" in tokenizer.get_vocab() else "<|en|>"
        prompt_tokens.append(lang_token)
        prompt_tokens.append(f"<|{task}|>")
        if without_timestamps:
            prompt_tokens.append("<|notimestamps|>")

        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        generated_ids: List[int] = []
        past_key_values = None

        # 3. Autoregressive Loop with KV Cache
        for step in range(max_new_tokens):
            if step == 0:
                # First pass: Process initial prompt sequence
                current_input = input_ids
            else:
                # Subsequent passes: Feed ONLY the latest predicted token
                current_input = input_ids[:, -1:]

            hidden_states, past_key_values = self.model.decoder(
                input_ids=current_input,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Project logits from the last position
            logits = self.proj_out(hidden_states[:, -1, :])

            # Greedy Next-Token Selection
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())

            if token_id == eos_token_id:
                break

            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return generated_ids