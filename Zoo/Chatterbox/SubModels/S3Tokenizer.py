from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2

from Zoo.Chatterbox.utils.audio_utils import load_audio_tensor, get_mel_basis, extract_power_spectrogram, N_FFT


# Audio Tokenizer Constants
S3_TOKEN_HOP = 640      # 25 tokens/sec (4 mel frames per token)
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


def pad_mel_batch(mels: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads a list of 2D mels [128, T_i] into a single 3D batch [B, 128, max_T]."""
    max_len = max(mel.shape[-1] for mel in mels)
    padded_mels, mel_lens = [], []

    for mel in mels:
        mel_lens.append(mel.shape[-1])
        pad_len = max_len - mel.shape[-1]
        if pad_len > 0:
            mel = F.pad(mel, (0, pad_len), mode="constant", value=pad_value)
        padded_mels.append(mel)

    return torch.stack(padded_mels, dim=0), torch.tensor(mel_lens, dtype=torch.long)


def drop_invalid_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Drops SOS, EOS, and keeps only valid speech codes (< 6561)."""
    return tokens[tokens < SPEECH_VOCAB_SIZE]


class S3Tokenizer(S3TokenizerV2):
    """S3Tokenizer wrapper converting 16kHz audio into 25Hz discrete speech tokens."""

    ignore_state_dict_missing = ("_mel_filters", "window")
    _mel_filters: torch.Tensor
    window: torch.Tensor

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: ModelConfig = ModelConfig()):
        super().__init__(name)
        
        # Native GPU Buffers using shared utils
        self.register_buffer("_mel_filters", get_mel_basis(config.n_mels))
        self.register_buffer("window", torch.hann_window(N_FFT))

    @property
    def device(self) -> torch.device:
        return self._mel_filters.device

    def log_mel_spectrogram(self, audio: torch.Tensor, padding: int = 0) -> torch.Tensor:
        """Computes 128-bin log-mel spectrogram [1, 128, T]."""
        audio = audio.to(self.device)
        magnitudes = extract_power_spectrogram(audio, self.window, padding)
        mel = self._mel_filters @ magnitudes

        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    @torch.no_grad()
    def forward(
        self,
        wavs: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes 16kHz audio into discrete tokens (25 tokens/sec)."""
        if isinstance(wavs, (torch.Tensor, np.ndarray)):
            if not torch.is_tensor(wavs):
                wavs = torch.from_numpy(wavs)
            wav_list = [w for w in (wavs if wavs.ndim > 1 else wavs.unsqueeze(0))]
        else:
            wav_list = [torch.from_numpy(w).float() if not torch.is_tensor(w) else w.float() for w in wavs]

        mels = []
        for wav in wav_list:
            mel = self.log_mel_spectrogram(wav.view(1, -1))
            if max_len is not None:
                mel = mel[..., : max_len * 4]
            mels.append(mel.squeeze(0))

        padded_mels, mel_lens = pad_mel_batch(mels)
        speech_tokens, speech_token_lens = self.quantize(
            padded_mels.to(self.device), mel_lens.to(self.device)
        )
        return speech_tokens.long().detach(), speech_token_lens.long().detach()

    @torch.inference_mode()
    def extract_prompt_tokens(
        self, wav_path_or_tensor: Union[str, torch.Tensor, np.ndarray], max_tokens: int = 375
    ) -> torch.Tensor:
        """Loads audio, resamples to 16kHz, and extracts prompt tokens [1, N]."""
        audio = load_audio_tensor(wav_path_or_tensor)
        tokens, _ = self.forward(audio, max_len=max_tokens)
        return tokens.to(self.device)