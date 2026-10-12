"""Unified Multimodal Preprocessor for Chatterbox Text-to-Speech & Voice Cloning."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from Zoo.Chatterbox.configs import ChatterboxConfig
from Zoo.Chatterbox.modules.MTLTokenizer import MTLTokenizer
from Zoo.Chatterbox.processing.language_normalization import normalize_punctuation


class ChatterboxProcessor:
    """Multimodal input pipeline handling prompt text normalization, language routing,

    and multi-rate reference audio preprocessing.
    """

    def __init__(
        self,
        tokenizer: Optional[MTLTokenizer] = None,
        config: Optional[ChatterboxConfig] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        cj_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or ChatterboxConfig()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif vocab_path is not None:
            self.tokenizer = MTLTokenizer(vocab_path=vocab_path, cj_path=cj_path)
        else:
            self.tokenizer = MTLTokenizer.from_pretrained()

    def process_text(
        self,
        text: str,
        language: str = "en",
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Cleans, normalizes, tokenizes, and wraps prompt text with SOT and EOT delimiters.

        Args:
            text: Raw input text.
            language: Target ISO language code (e.g., 'en', 'es', 'zh', 'ja').
            device: Target device for output tensor.

        Returns:
            LongTensor of shape (1, seq_len + 2) padded with start and stop tokens.
        """
        clean_text = normalize_punctuation(text)
        tokens = self.tokenizer.text_to_tokens(clean_text, lang=language.lower(), device=device)

        sot = self.config.t3_config.start_text_token
        eot = self.config.t3_config.stop_text_token

        # Prepend SOT (255) and append EOT (0)
        tokens = F.pad(tokens, (1, 0), value=sot)
        tokens = F.pad(tokens, (0, 1), value=eot)
        return tokens

    def process_audio(
        self,
        audio_path_or_array: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and standardizes reference voice samples into synchronized 24 kHz and 16 kHz signals.

        Args:
            audio_path_or_array: File path string or raw waveform array/tensor.

        Returns:
            ref_24k: Float32 mono array at 24000 Hz.
            ref_16k: Float32 mono array at 16000 Hz.
        """
        sr_24k = self.config.sample_rate_24k
        sr_16k = self.config.sample_rate_16k

        if isinstance(audio_path_or_array, (str, Path)):
            wav_24k, _ = librosa.load(str(audio_path_or_array), sr=sr_24k, mono=True)
        elif isinstance(audio_path_or_array, torch.Tensor):
            wav_np = audio_path_or_array.detach().cpu().float().numpy()
            if wav_np.ndim > 1:
                wav_np = np.mean(wav_np, axis=0)
            wav_24k = wav_np
        elif isinstance(audio_path_or_array, np.ndarray):
            wav_24k = audio_path_or_array
            if wav_24k.ndim > 1:
                wav_24k = np.mean(wav_24k, axis=0)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_path_or_array)}")

        wav_24k = np.asarray(wav_24k, dtype=np.float32)
        wav_16k = librosa.resample(wav_24k, orig_sr=sr_24k, target_sr=sr_16k)

        return wav_24k, wav_16k

    def __call__(
        self,
        text: str,
        reference_audio: Union[str, Path, np.ndarray, torch.Tensor],
        language: str = "en",
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Unified preprocessing callable."""
        text_tokens = self.process_text(text, language=language, device=device)
        ref_24k, ref_16k = self.process_audio(reference_audio)

        return {
            "text_tokens": text_tokens,
            "ref_24k": ref_24k,
            "ref_16k": ref_16k,
            "language": language,
        }