import torch
import torch.nn.functional as F
import torchaudio

class WhisperFeatureExtractor:
    """
    Turns raw audio waveforms into Log-Mel Spectrograms.
    This class performs pure signal processing (no learnable weights).
    """
    def __init__(
        self, 
        feature_size: int = 128, 
        sampling_rate: int = 16000, 
        hop_length: int = 160, # How far the window slides forward before taking the next snapshot. each window overlaps the previous one by 240 samples (60 %)
        chunk_length: int = 30, 
        n_fft: int = 400 # Window Size: The length of each analyzed audio slice.
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        
        # Whisper always processes exactly 30 seconds of audio at a time
        self.n_samples = chunk_length * sampling_rate  # 30 * 16000 = 480,000 samples

        # Generate the Mel Filterbank matrix. Output shape: (201, 80)
        self.mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            f_min=0.0,
            f_max=8000.0,
            n_mels=feature_size,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney"
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: torch.Tensor of shape (Batch, Seq_Len) or (Seq_Len,)
                      Must be sampled at 16,000 Hz.
        Returns:
            log_spec: torch.Tensor of shape (Batch, 80, 3000)
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
            waveform = F.pad(waveform, (0, pad_amount), value=0.0)

        # 3. Compute Short-Time Fourier Transform (STFT)
        window = torch.hann_window(self.n_fft, device=device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )

        # 4. Get Magnitudes (discard the last frame to perfectly match HF Whisper)
        # stft shape before: (Batch, 201, 3001) -> after: (Batch, 201, 3000)
        magnitudes = (stft[..., :-1].abs() ** 2)

        # 5. Project to Mel Scale
        mel_filters = self.mel_filters.to(device)
        # Transpose mel_filters to (80, 201) and multiply by magnitudes (Batch, 201, 3000)
        # Resulting shape: (Batch, 80, 3000)
        mel_spec = torch.matmul(mel_filters.T, magnitudes)

        # 6. Log10 and Clamping (Dynamic Range Compression)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        
        # 7. Normalize values strictly matching Hugging Face's formula
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec