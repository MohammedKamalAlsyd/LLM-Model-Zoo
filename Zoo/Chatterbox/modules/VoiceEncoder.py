"""3-Layer LSTM Speaker Encoder for T3 conditioning.

Extracts 256-dimensional L2-normalized d-vectors from 16 kHz speech.
"""

from typing import List, Optional, Union
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Zoo.Chatterbox.configs import VoiceEncoderConfig


def extract_ve_mel_spectrogram(
    audio: torch.Tensor,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    cfg: VoiceEncoderConfig,
) -> torch.Tensor:
    """Computes Mel spectrogram matching the exact numerical behavior of Chatterbox VoiceEncoder."""
    if cfg.preemphasis > 0.0:
        audio = torch.cat(
            [audio[..., :1], audio[..., 1:] - cfg.preemphasis * audio[..., :-1]],
            dim=-1,
        ).clamp(-1.0, 1.0)

    audio_tensor = audio if audio.ndim == 2 else audio.unsqueeze(0)
    mag = torch.stft(
        audio_tensor,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=window.to(audio.device),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ).abs()

    if cfg.mel_power != 1.0:
        mag = mag.pow(cfg.mel_power)

    mel = torch.matmul(mel_filters.to(audio.device).unsqueeze(0), mag)

    if cfg.mel_type == "db":
        mel = 20.0 * torch.log10(torch.clamp(mel, min=cfg.stft_magnitude_min))

    if cfg.normalized_mels:
        min_db = 20.0 * np.log10(cfg.stft_magnitude_min)
        mel = (mel - min_db) / (-min_db + 15.0)

    return mel


class VoiceEncoder(nn.Module):
    """3-Layer LSTM Speaker Encoder projecting partial spectrogram windows into speaker embeddings."""

    _mel_filters: torch.Tensor
    window: torch.Tensor

    def __init__(self, config: Optional[VoiceEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or VoiceEncoderConfig()

        self.lstm = nn.LSTM(
            input_size=self.config.num_mels,
            hidden_size=self.config.ve_hidden_size,
            num_layers=3,
            batch_first=True,
        )
        if self.config.flatten_lstm_params:
            self.lstm.flatten_parameters()

        self.proj = nn.Linear(self.config.ve_hidden_size, self.config.speaker_embed_size)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

        mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.num_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        self.register_buffer("_mel_filters", torch.from_numpy(mel_basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(self.config.n_fft, periodic=False), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """Args:

            mels: Tensor of shape (batch_size, frames, num_mels).

        Returns:
            Normalized speaker embeddings of shape (batch_size, speaker_embed_size).
        """
        _, (hidden, _) = self.lstm(mels)
        last_hidden = hidden[-1]
        embeds = F.relu(self.proj(last_hidden)) if self.config.ve_final_relu else self.proj(last_hidden)
        return embeds / torch.linalg.norm(embeds, dim=1, keepdim=True)

    def inference(
        self,
        mels: torch.Tensor,
        mel_lens: Union[torch.Tensor, List[int]],
        overlap: float = 0.5,
        rate: Optional[float] = None,
        min_coverage: float = 0.8,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Splits full utterance spectrograms into overlapping chunks and computes average embeddings."""
        resolved_mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens
        part_frames = self.config.ve_partial_frames

        if rate is not None and rate > 0:
            f_step = int(np.round((self.config.sample_rate / rate) / part_frames))
        else:
            f_step = int(np.round(part_frames * (1.0 - overlap)))

        partials: List[torch.Tensor] = []
        n_partials: List[int] = []

        for mel, l in zip(mels, resolved_mel_lens):
            n_wins, rem = divmod(max(l - part_frames + f_step, 0), f_step)
            if n_wins == 0 or (rem + (part_frames - f_step)) / part_frames >= min_coverage:
                n_wins += 1
            n_partials.append(n_wins)

            target_len = part_frames + f_step * (n_wins - 1)
            padded_mel = F.pad(mel, (0, 0, 0, max(0, target_len - mel.size(0))))[:target_len]
            partials.extend([padded_mel[i * f_step : i * f_step + part_frames] for i in range(n_wins)])

        stacked_partials = torch.stack(partials)
        total_chunks = max(1, int(np.ceil(len(stacked_partials) / (batch_size or len(stacked_partials)))))
        p_embeds = torch.cat([self(b) for b in stacked_partials.chunk(total_chunks)], dim=0).cpu()

        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = torch.stack([p_embeds[s:e].mean(dim=0) for s, e in zip(slices[:-1], slices[1:])])
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate: int,
        as_spk: bool = False,
        batch_size: int = 32,
        trim_top_db: Optional[float] = 20.0,
        **kwargs,
    ) -> np.ndarray:
        """Extracts utterance or speaker-averaged embeddings directly from raw numpy waveforms."""
        mels: List[np.ndarray] = []
        for w in wavs:
            if sample_rate != self.config.sample_rate:
                w = librosa.resample(w, orig_sr=sample_rate, target_sr=self.config.sample_rate, res_type="kaiser_fast")
            if trim_top_db is not None:
                w = librosa.effects.trim(w, top_db=trim_top_db)[0]

            w_tensor = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(self.device)
            mel = extract_ve_mel_spectrogram(w_tensor, self._mel_filters, self.window, self.config)
            mels.append(mel.squeeze(0).T.cpu().numpy())

        kwargs.setdefault("rate", 1.3)
        mel_lens = [m.shape[0] for m in mels]
        max_len = max(mel_lens)
        packed = torch.zeros(len(mels), max_len, mels[0].shape[1])
        for i, m in enumerate(mels):
            packed[i, :m.shape[0]] = torch.from_numpy(np.asarray(m))

        with torch.inference_mode():
            utt_embeds = self.inference(packed.to(self.device), mel_lens, batch_size=batch_size, **kwargs).numpy()

        if as_spk:
            spk_emb = np.mean(utt_embeds, axis=0)
            return spk_emb / np.linalg.norm(spk_emb, 2)
        return utt_embeds