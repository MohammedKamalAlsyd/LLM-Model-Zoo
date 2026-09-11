from typing import List, Optional, Tuple
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import S3TokenizerV2, ModelConfig

S3_SR, S3_HOP, S3_TOKEN_HOP, S3_TOKEN_RATE, SPEECH_VOCAB_SIZE = 16_000, 160, 640, 25, 6561


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """
    window: torch.Tensor
    _mel_filters: torch.Tensor

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: ModelConfig = ModelConfig()):
        super().__init__(name)
        self.n_fft = 400
        mel = librosa.filters.mel(sr=S3_SR, n_fft=self.n_fft, n_mels=config.n_mels)
        self.register_buffer("_mel_filters", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length is multiple of 40ms (S3 runs at 25 token/sec).
        """
        step = sr / S3_TOKEN_RATE
        return [
            F.pad(w, (0, int(np.ceil(w.shape[-1] / step) * step) - w.shape[-1]))
            for w in self._prepare_audio(wavs)
        ]

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        return [torch.atleast_2d(torch.as_tensor(w)) for w in wavs]

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor` and handles a list of wavs one by one, which is unexpected.

        Args
        ----
        - `wavs`: 16 kHz speech audio
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        mels = [
            self.log_mel_spectrogram(w.to(self.device))[..., : max_len * 4 if max_len else None].squeeze(0)
            for w in self._prepare_audio(wavs)
        ]
        mels, mel_lens = padding(mels)
        speech_tokens, speech_token_lens = self.quantize(mels, mel_lens.to(self.device))
        return speech_tokens.long().detach(), speech_token_lens.detach()

    def log_mel_spectrogram(self, audio: torch.Tensor, padding: int = 0):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        audio = torch.as_tensor(audio, device=self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        stft = torch.stft(audio, self.n_fft, S3_HOP, window=self.window.to(self.device), return_complex=True)
        mel_spec = self._mel_filters.to(self.device) @ stft[..., :-1].abs().pow(2)

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0