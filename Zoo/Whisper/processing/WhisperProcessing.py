"""Pure Signal Processing Audio Feature Extractor for Whisper (No Learnable Weights)."""

from typing import Optional, Union
import torch
import torch.nn.functional as F
import torchaudio

from Zoo.Whisper.configs import WhisperAudioConfig


class WhisperFeatureExtractor:
    """Converts 16kHz audio waveforms into normalized 128-bin Log-Mel Spectrograms."""

    def __init__(self, config: Optional[WhisperAudioConfig] = None) -> None:
        self.config = config or WhisperAudioConfig()
        self.feature_size = self.config.feature_size
        self.sampling_rate = self.config.sampling_rate
        self.hop_length = self.config.hop_length
        self.chunk_length = self.config.chunk_length
        self.n_fft = self.config.n_fft
        self.n_samples = self.config.n_samples

        # Slaney Mel Filterbank matrix of shape: (201, feature_size)
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=int(self.n_fft // 2 + 1),
            f_min=0.0,
            f_max=8000.0,
            n_mels=self.feature_size,
            sample_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_mel_filters = mel_filters

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts Log-Mel Spectrogram matching Hugging Face Whisper preprocessing.

        Args:
            waveform: Tensor of shape (Batch, Samples) or (Samples,). Sample rate must be 16kHz.

        Returns:
            log_spec: Tensor of shape (Batch, 128, 3000).
        """
        device = waveform.device

        # 1. Ensure Batch Dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 2. Pad or Trim to exactly 30 seconds (480,000 samples)
        seq_len = waveform.shape[-1]
        if seq_len > self.n_samples:
            waveform = waveform[..., :self.n_samples]
        elif seq_len < self.n_samples:
            pad_amount = self.n_samples - seq_len
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0.0)

        # 3. Short-Time Fourier Transform (STFT) with Hann Window
        window = torch.hann_window(self.n_fft, device=device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )

        # 4. Magnitudes squared (discarding the last frame to yield exactly 3000 frames)
        magnitudes = stft[..., :-1].abs() ** 2

        # 5. Project onto 128 Mel Filterbanks
        mel_filters = self.register_mel_filters.to(device=device, dtype=magnitudes.dtype)
        mel_spec = torch.matmul(mel_filters.T, magnitudes)

        # 6. Dynamic Range Compression (Log10 scale)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # 7. Slaney/Hugging Face Normalization formula
        max_val = log_spec.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec