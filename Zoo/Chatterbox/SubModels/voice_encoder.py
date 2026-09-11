from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from Zoo.Chatterbox.utils.audio_utils import extract_mel_spectrogram

@dataclass(frozen=True) 
class VoiceEncConfig:
    num_mels: int = 40
    sample_rate: int = 16000
    speaker_embed_size: int = 256
    ve_hidden_size: int = 256
    flatten_lstm_params: bool = False
    n_fft: int = 400
    hop_size: int = 160
    win_size: int = 400
    fmax: int = 8000
    fmin: int = 0
    preemphasis: float = 0.0
    mel_power: float = 2.0
    mel_type: str = "amp"
    normalized_mels: bool = False
    ve_partial_frames: int = 160
    ve_final_relu: bool = True
    stft_magnitude_min: float = 1e-4

class VoiceEncoder(nn.Module):
    """3-Layer LSTM Speaker Encoder. Extracts 256-dim L2-normalized speaker embeddings."""
    _mel_filters: torch.Tensor
    window: torch.Tensor

    def __init__(self, hp: VoiceEncConfig = VoiceEncConfig()):
        """Initializes the network layers, projection heads, similarity parameters, and cached GPU buffers."""
        super().__init__()
        self.hp = hp
        self.lstm = nn.LSTM(hp.num_mels, hp.ve_hidden_size, num_layers=3, batch_first=True)
        if hp.flatten_lstm_params: self.lstm.flatten_parameters()
        self.proj = nn.Linear(hp.ve_hidden_size, hp.speaker_embed_size)

        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

        mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)
        self.register_buffer("_mel_filters", torch.from_numpy(mel_basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(hp.n_fft, periodic=False), persistent=False)

    @property
    def device(self) -> torch.device:
        """Returns the device (CPU/GPU) where the model parameters reside."""
        return next(self.parameters()).device

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """Computes L2-normalized speaker embeddings for a batch of fixed-duration partial mel windows."""
        _, (hidden, _) = self.lstm(mels)
        embeds = F.relu(self.proj(hidden[-1])) if self.hp.ve_final_relu else self.proj(hidden[-1])
        return embeds / torch.linalg.norm(embeds, dim=1, keepdim=True)

    def inference(self, mels, mel_lens, overlap=0.5, rate=None, min_coverage=0.8, batch_size=None):
        """Segments continuous utterance mels into overlapping partial windows, runs inference, and averages them."""
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens
        f_step = int(np.round((self.hp.sample_rate / rate) / self.hp.ve_partial_frames)) if rate else int(np.round(self.hp.ve_partial_frames * (1 - overlap)))
        
        partials, n_partials = [], []
        for mel, l in zip(mels, mel_lens):
            n_wins, rem = divmod(max(l - self.hp.ve_partial_frames + f_step, 0), f_step)
            if n_wins == 0 or (rem + (self.hp.ve_partial_frames - f_step)) / self.hp.ve_partial_frames >= min_coverage: 
                n_wins += 1
            n_partials.append(n_wins)
            
            target_len = self.hp.ve_partial_frames + f_step * (n_wins - 1)
            mel = F.pad(mel, (0, 0, 0, max(0, target_len - mel.size(0))))[:target_len]
            partials.extend([mel[i * f_step : i * f_step + self.hp.ve_partial_frames] for i in range(n_wins)])
        
        partials = torch.stack(partials)
        chunks = max(1, int(np.ceil(len(partials) / (batch_size or len(partials)))))
        p_embeds = torch.cat([self(b) for b in partials.chunk(chunks)], dim=0).cpu()
        
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = torch.stack([p_embeds[s:e].mean(0) for s, e in zip(slices[:-1], slices[1:])])
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def embeds_from_wavs(
            self,
            wavs: List[np.ndarray],
            sample_rate: int,
            as_spk: bool = False,
            batch_size: int = 32,
            trim_top_db: Optional[float] = 20,
            **kwargs,
        ) -> np.ndarray:
            """High-level pipeline wrapper that resamples, trims, extracts mel spectrograms, infers utterance embeddings, and optionally aggregates them into a single speaker embedding."""
            mels = []
            for w in wavs:
                if sample_rate != self.hp.sample_rate:
                    w = librosa.resample(w, orig_sr=sample_rate, target_sr=self.hp.sample_rate, res_type="kaiser_fast")
                if trim_top_db: 
                    w = librosa.effects.trim(w, top_db=trim_top_db)[0]
                
                mel = extract_mel_spectrogram(torch.from_numpy(np.asarray(w, dtype=np.float32)).to(self.device), self._mel_filters, self.window, self.hp)
                mels.append(mel.squeeze(0).T.cpu().numpy())
                
            kwargs.setdefault("rate", 1.3)
            
            mel_lens = [m.shape[0] for m in mels]
            packed = torch.zeros(len(mels), max(mel_lens), mels[0].shape[1])
            for i, m in enumerate(mels): 
                packed[i, :m.shape[0]] = torch.from_numpy(np.asarray(m))
            
            with torch.inference_mode():
                utt_embeds = self.inference(packed.to(self.device), mel_lens, batch_size=batch_size, **kwargs).numpy()
                
            if as_spk:
                emb = np.mean(utt_embeds, axis=0)
                return emb / np.linalg.norm(emb, 2)
            return utt_embeds