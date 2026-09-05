from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Chatterbox.utils.audio_utils import load_audio_tensor, get_mel_basis, extract_power_spectrogram, N_FFT


@dataclass(frozen=True) 
class VoiceEncConfig:
    num_mels: int = 40
    speaker_embed_size: int = 256
    ve_hidden_size: int = 256
    fmax: int = 8000
    fmin: int = 0
    ve_partial_frames: int = 160
    ve_final_relu: bool = True
    sample_rate: int = 16000


class VoiceEncoder(nn.Module):
    """3-Layer LSTM Speaker Encoder. Extracts 256-dim L2-normalized speaker embedding."""
    _mel_filters: torch.Tensor
    window: torch.Tensor
    
    def __init__(self, hp: VoiceEncConfig = VoiceEncConfig()):
        super().__init__()
        self.hp = hp

        self.lstm = nn.LSTM(hp.num_mels, hp.ve_hidden_size, num_layers=3, batch_first=True)
        self.proj = nn.Linear(hp.ve_hidden_size, hp.speaker_embed_size)

        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

        # Native GPU Buffers replace the old @lru_cache and librosa logic!
        self.register_buffer("_mel_filters", get_mel_basis(hp.num_mels, hp.fmin, hp.fmax), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, partial_mels: torch.Tensor) -> torch.Tensor:
        """Embeds a batch of fixed-duration partial windows."""
        _, (hidden, _) = self.lstm(partial_mels)
        embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            embeds = F.relu(embeds)
        return embeds / torch.linalg.norm(embeds, dim=1, keepdim=True)

    def _segment_mel(self, mel: torch.Tensor, frame_step: int) -> torch.Tensor:
        """Segments a continuous mel [T, 40] into overlapping partial windows [N, 160, 40]."""
        n_frames = mel.size(0)
        win = self.hp.ve_partial_frames

        if n_frames < win:
            mel = F.pad(mel, (0, 0, 0, win - n_frames))
            n_frames = win

        n_partials = max(1, int(np.ceil((n_frames - win) / frame_step)) + 1)
        target_len = win + (n_partials - 1) * frame_step

        if target_len > n_frames:
            mel = F.pad(mel, (0, 0, 0, target_len - n_frames))

        partials = [mel[i * frame_step : i * frame_step + win] for i in range(n_partials)]
        return torch.stack(partials)

    @torch.inference_mode()
    def embed_utterance(self, mel: torch.Tensor, rate: float = 1.3) -> torch.Tensor:
        """Derives a single 256-dim speaker embedding by averaging sliding windows."""
        frame_step = max(1, int(np.round((self.hp.sample_rate / rate) / self.hp.ve_partial_frames)))
        partials = self._segment_mel(mel, frame_step).to(self.device)

        partial_embeds = self(partials)
        mean_embed = torch.mean(partial_embeds, dim=0)
        return mean_embed / torch.linalg.norm(mean_embed, dim=0, keepdim=True)

    @torch.inference_mode()
    def extract_speaker_embedding(
        self, wav_or_path: Union[str, np.ndarray, torch.Tensor], trim_top_db: Optional[float] = 20.0
    ) -> torch.Tensor:
        """High-level inference: Loads/cleans audio and returns a [256] embedding on GPU."""
        audio = load_audio_tensor(wav_or_path, trim_db=trim_top_db).to(self.device)
        
        # Fast PyTorch GPU Mel Extraction
        magnitudes = extract_power_spectrogram(audio, self.window)
        mel = (self._mel_filters @ magnitudes).squeeze(0).T  # Transpose to [T, 40]
        
        return self.embed_utterance(mel)