"""S3 Speech Tokenizer (25 Hz discrete acoustic quantizer).

Computes 16 kHz Log-Mel spectrograms with exact numerical clamp bounds.
"""

from typing import List, Optional, Tuple, Union
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2
from s3tokenizer.utils import padding

S3_SR: int = 16000
S3_HOP: int = 160
S3_TOKEN_HOP: int = 640
S3_TOKEN_RATE: int = 25
SPEECH_VOCAB_SIZE: int = 6561


class S3Tokenizer(S3TokenizerV2):
    """Subclass of S3TokenizerV2 preserving registered FFT buffers and exact scaling."""

    window: torch.Tensor
    _mel_filters: torch.Tensor

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: Optional[ModelConfig] = None) -> None:
        cfg = config or ModelConfig()
        super().__init__(name)
        self.n_fft = 400
        mel_basis = librosa.filters.mel(sr=S3_SR, n_fft=self.n_fft, n_mels=cfg.n_mels)
        self.register_buffer("_mel_filters", torch.from_numpy(mel_basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def pad(self, wavs: List[Union[np.ndarray, torch.Tensor]], sr: int) -> List[torch.Tensor]:
        """Pads arbitrary waveforms to the nearest 40 ms boundary (1 token = 40 ms at 25 Hz)."""
        step = sr / S3_TOKEN_RATE
        return [
            F.pad(w, (0, int(np.ceil(w.shape[-1] / step) * step) - w.shape[-1]))
            for w in self._prepare_audio(wavs)
        ]

    def _prepare_audio(self, wavs: Union[torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]) -> List[torch.Tensor]:
        if isinstance(wavs, torch.Tensor) and wavs.ndim == 2:
            return [wavs[i : i + 1] for i in range(wavs.size(0))]
        return [torch.atleast_2d(torch.as_tensor(w)) for w in wavs]

    def log_mel_spectrogram(self, audio: torch.Tensor, padding_len: int = 0) -> torch.Tensor:
        r"""Computes 128-bin log-Mel spectrogram matching the pretrained S3Tokenizer checkpoint.

        $$\text{mel} = \mathbf{M} \cdot |\text{STFT}(x)|^2$$
        $$\text{spec} = \frac{\max(\log_{10}(\max(\text{mel}, 10^{-10})), \max(\text{mel}) - 8.0) + 4.0}{4.0}$$
        """
        audio = torch.as_tensor(audio, device=self.device)
        if padding_len > 0:
            audio = F.pad(audio, (0, padding_len))

        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window.to(self.device),
            return_complex=True,
        )
        # Power spectrogram
        mel_spec = self._mel_filters.to(self.device) @ stft[..., :-1].abs().pow(2)

        # Clamped logarithmic compression
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    @torch.no_grad()
    def forward(
        self,
        wavs: Union[torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            wavs: 16 kHz audio waveforms.
            max_len: Optional truncation length in tokens (25 tokens per second).

        Returns:
            speech_tokens: LongTensor of shape (batch_size, num_tokens).
            speech_token_lens: LongTensor of valid lengths per item.
        """
        mels = [
            self.log_mel_spectrogram(w.to(self.device))[..., : max_len * 4 if max_len else None].squeeze(0)
            for w in self._prepare_audio(wavs)
        ]
        padded_mels, mel_lens = padding(mels)
        speech_tokens, speech_token_lens = self.quantize(padded_mels, mel_lens.to(self.device))
        return speech_tokens.long().detach(), speech_token_lens.detach()