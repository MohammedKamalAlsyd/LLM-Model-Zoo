from typing import Optional, Union
import librosa
import numpy as np
import torch
import torch.nn.functional as F

# Shared Chatterbox Audio Constants
SAMPLE_RATE = 16_000
N_FFT = 400
HOP_SIZE = 160


def load_audio_tensor(wav_input: Union[str, np.ndarray, torch.Tensor], trim_db: Optional[float] = None) -> torch.Tensor:
    """Loads audio from path/array/tensor and standardizes to a 16kHz 1D PyTorch Tensor."""
    if isinstance(wav_input, str):
        wav, _ = librosa.load(wav_input, sr=SAMPLE_RATE)
    elif isinstance(wav_input, torch.Tensor):
        wav = wav_input.squeeze().cpu().numpy()
    else:
        wav = np.asarray(wav_input, dtype=np.float32).squeeze()
        
    if trim_db is not None:
        wav, _ = librosa.effects.trim(wav, top_db=trim_db)
        
    return torch.from_numpy(wav).float()


def get_mel_basis(n_mels: int, fmin: int = 0, fmax: int = 8000) -> torch.Tensor:
    """Precomputes the Mel filterbank matrix."""
    filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(filters).float()


def extract_power_spectrogram(audio: torch.Tensor, window: torch.Tensor, padding: int = 0) -> torch.Tensor:
    """Computes STFT power magnitudes [B, F, T]. Matches librosa.stft exactly but runs on GPU."""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
        
    stft = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_SIZE,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True
    )
    return stft[..., :-1].abs() ** 2.0