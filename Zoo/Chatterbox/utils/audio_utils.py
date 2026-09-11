import numpy as np
import torch

def extract_mel_spectrogram(audio: torch.Tensor, mel_filters: torch.Tensor, window: torch.Tensor, hp) -> torch.Tensor:
    """Computes Mel spectrogram exactly matching original Chatterbox librosa configuration."""
    if hp.preemphasis > 0:
        audio = torch.cat([audio[..., :1], audio[..., 1:] - hp.preemphasis * audio[..., :-1]], dim=-1).clamp(-1, 1)

    mag = torch.stft(audio if audio.ndim == 2 else audio.unsqueeze(0), n_fft=hp.n_fft, 
                     hop_length=hp.hop_size, win_length=hp.win_size, window=window.to(audio.device), 
                     center=True, pad_mode="reflect", return_complex=True).abs()

    if hp.mel_power != 1.0: 
        mag = mag.pow(hp.mel_power)
        
    mel = torch.matmul(mel_filters.to(audio.device).unsqueeze(0), mag)

    if hp.mel_type == "db": 
        mel = 20.0 * torch.log10(torch.clamp(mel, min=hp.stft_magnitude_min))
        
    if hp.normalized_mels:
        min_db = 20.0 * np.log10(hp.stft_magnitude_min)
        mel = (mel - min_db) / (-min_db + 15.0)
        
    return mel