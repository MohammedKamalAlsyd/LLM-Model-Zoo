"""Universal Audio Utilities for Speech Foundation Models (Whisper, AudioLDM, MMS).

Provides robust signal loading, streaming resampling, long-form chunking with overlap,
and timestamp formatting.
"""

from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T


def load_and_resample_audio(
    audio_path_or_array: Union[str, np.ndarray, torch.Tensor],
    target_sample_rate: int = 16000,
    orig_sample_rate: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Loads arbitrary audio inputs, converts to mono, and standardizes to the target sample rate.

    Args:
        audio_path_or_array: File path string, numpy array, or torch tensor.
        target_sample_rate: Standard sample rate required by the model (default 16000 Hz).
        orig_sample_rate: Required if passing a raw numpy array or tensor without a path.

    Returns:
        waveform: Float32 Tensor of shape (1, num_samples).
        sample_rate: Integer sample rate (always target_sample_rate).
    """
    if isinstance(audio_path_or_array, str):
        data, sr = sf.read(audio_path_or_array, dtype="float32")
        waveform = torch.from_numpy(data)
        orig_sample_rate = sr
    elif isinstance(audio_path_or_array, np.ndarray):
        waveform = torch.from_numpy(audio_path_or_array).float()
        if orig_sample_rate is None:
            raise ValueError("orig_sample_rate must be specified when passing numpy array.")
    elif isinstance(audio_path_or_array, torch.Tensor):
        waveform = audio_path_or_array.float()
        if orig_sample_rate is None:
            raise ValueError("orig_sample_rate must be specified when passing torch.Tensor.")
    else:
        raise TypeError(f"Unsupported audio type: {type(audio_path_or_array)}")

    # Standardize shape to (Channels, Samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.transpose(0, 1)

    # Convert Multi-channel / Stereo to Mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if sample rate does not match
    if orig_sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform, target_sample_rate


def chunk_long_audio(
    waveform: torch.Tensor,
    chunk_length_seconds: float = 30.0,
    stride_seconds: float = 25.0,
    sample_rate: int = 16000,
) -> Generator[Tuple[torch.Tensor, float, float], None, None]:
    """Generates sequential chunks of audio with overlap to transcribe arbitrarily long files.

    Args:
        waveform: Mono tensor of shape (1, total_samples).
        chunk_length_seconds: Window size per inference pass (default 30.0s for Whisper).
        stride_seconds: Stride / step size before extracting the next window (default 25.0s, giving 5s overlap).
        sample_rate: Audio sample rate (default 16000 Hz).

    Yields:
        chunk: Tensor of shape (1, chunk_samples).
        start_time_seconds: Float start time timestamp.
        end_time_seconds: Float end time timestamp.
    """
    total_samples = waveform.shape[-1]
    chunk_samples = int(chunk_length_seconds * sample_rate)
    stride_samples = int(stride_seconds * sample_rate)

    if total_samples <= chunk_samples:
        yield waveform, 0.0, total_samples / sample_rate
        return

    current_idx = 0
    while current_idx < total_samples:
        end_idx = min(current_idx + chunk_samples, total_samples)
        chunk = waveform[:, current_idx:end_idx]
        
        start_time = current_idx / sample_rate
        end_time = end_idx / sample_rate
        
        yield chunk, start_time, end_time
        
        if end_idx == total_samples:
            break
        current_idx += stride_samples


def format_timestamp(seconds: float) -> str:
    """Formats raw seconds into an SRT/VTT timestamp string (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"