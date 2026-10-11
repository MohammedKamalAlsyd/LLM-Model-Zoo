"""Configuration dataclasses for Whisper Audio Encoder, Text Decoder, and Generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class WhisperAudioConfig:
    """Signal processing parameters for log-mel spectrogram extraction."""
    feature_size: int = 128          # 128 mel filters for large-v3/turbo (80 in v1/v2)
    sampling_rate: int = 16000       # 16 kHz
    hop_length: int = 160            # 10ms hop
    chunk_length: int = 30           # 30-second context window
    n_fft: int = 400                 # 25ms window size (400 samples)
    n_samples: int = 480000          # 30 * 16000 samples
    nb_max_frames: int = 3000        # Spectrogram time frames (480000 / 160)


@dataclass
class WhisperEncoderConfig:
    """Hyperparameters for the Audio Transformer Encoder."""
    d_model: int = 1280
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    max_source_positions: int = 1500  # Stride-2 conv shrinks 3000 -> 1500


@dataclass
class WhisperDecoderConfig:
    """Hyperparameters for the Text Transformer Decoder."""
    d_model: int = 1280
    vocab_size: int = 51866
    decoder_layers: int = 4           # TURBO: 4 layers (Standard large-v3 uses 32)
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    max_target_positions: int = 448   # Maximum tokens produced per 30-second segment


@dataclass
class WhisperConfig:
    """Unified configuration matching openai/whisper-large-v3-turbo."""
    audio_config: WhisperAudioConfig = field(default_factory=WhisperAudioConfig)
    encoder_config: WhisperEncoderConfig = field(default_factory=WhisperEncoderConfig)
    decoder_config: WhisperDecoderConfig = field(default_factory=WhisperDecoderConfig)

    @classmethod
    def from_pretrained_name(cls, model_name: str = "openai/whisper-large-v3-turbo") -> "WhisperConfig":
        """Instantiates default configuration for whisper-large-v3-turbo."""
        if "turbo" in model_name:
            dec_cfg = WhisperDecoderConfig(decoder_layers=4)
        else:
            dec_cfg = WhisperDecoderConfig(decoder_layers=32)
        return cls(decoder_config=dec_cfg)