"""Consolidated configuration dataclasses for Chatterbox V3 Multilingual TTS."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class VoiceEncoderConfig:
    """Configuration for the 3-Layer LSTM Speaker Encoder."""
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VoiceEncoderConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_keys})


@dataclass
class T3Config:
    """Configuration for the T3 Autoregressive Transformer Backbone."""
    text_tokens_dict_size: int = 2454
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_tokens: int = 2048

    speech_tokens_dict_size: int = 8194
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    max_speech_tokens: int = 4096

    llama_config_name: str = "Llama_520M"
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    input_pos_emb: str = "learned"
    speaker_embed_size: int = 256
    speech_cond_prompt_len: int = 150
    emotion_adv: bool = True
    perceiver_query_tokens: int = 32
    perceiver_num_heads: int = 4

    @property
    def n_channels(self) -> int:
        return self.hidden_size

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "T3Config":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_keys})


@dataclass
class S3FlowConfig:
    """Configuration for CAMPPlus, Conformer, and Flow Matching U-Net."""
    vocab_size: int = 6561
    input_size: int = 512
    output_size: int = 80
    spk_embed_dim: int = 192
    conformer_heads: int = 8
    conformer_linear_units: int = 2048
    conformer_num_blocks: int = 6
    conformer_up_blocks: int = 4
    conformer_dropout: float = 0.1
    token_mel_ratio: int = 2
    pre_lookahead_len: int = 3
    unet_channels: List[int] = field(default_factory=lambda: [256])
    unet_num_mid_blocks: int = 12
    unet_attention_head_dim: int = 64
    unet_n_blocks: int = 4
    unet_num_heads: int = 8
    act_fn: str = "gelu"
    causal: bool = True
    inference_cfg_rate: float = 0.7
    t_scheduler: str = "cosine"
    default_cfm_steps: int = 10

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "S3FlowConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_keys})


@dataclass
class S3VocoderConfig:
    """Configuration for HiFT-Net Neural Source Filter (NSF) Vocoder."""
    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 11, 7])
    istft_params: Dict[str, int] = field(default_factory=lambda: {"n_fft": 16, "hop_len": 4})
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    source_resblock_kernel_sizes: List[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "S3VocoderConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_keys})


@dataclass
class ChatterboxConfig:
    """Root configuration for end-to-end Chatterbox TTS pipeline."""
    voice_encoder_config: VoiceEncoderConfig = field(default_factory=VoiceEncoderConfig)
    t3_config: T3Config = field(default_factory=T3Config)
    s3_flow_config: S3FlowConfig = field(default_factory=S3FlowConfig)
    s3_vocoder_config: S3VocoderConfig = field(default_factory=S3VocoderConfig)
    sample_rate_16k: int = 16000
    sample_rate_24k: int = 24000
    speech_token_rate: int = 25
    max_enc_cond_len_sec: float = 6.0
    max_dec_cond_len_sec: float = 10.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChatterboxConfig":
        ve_dict = config_dict.get("voice_encoder_config", {})
        t3_dict = config_dict.get("t3_config", {})
        flow_dict = config_dict.get("s3_flow_config", {})
        vocoder_dict = config_dict.get("s3_vocoder_config", {})

        return cls(
            voice_encoder_config=VoiceEncoderConfig.from_dict(ve_dict) if isinstance(ve_dict, dict) else ve_dict,
            t3_config=T3Config.from_dict(t3_dict) if isinstance(t3_dict, dict) else t3_dict,
            s3_flow_config=S3FlowConfig.from_dict(flow_dict) if isinstance(flow_dict, dict) else flow_dict,
            s3_vocoder_config=S3VocoderConfig.from_dict(vocoder_dict) if isinstance(vocoder_dict, dict) else vocoder_dict,
            sample_rate_16k=config_dict.get("sample_rate_16k", 16000),
            sample_rate_24k=config_dict.get("sample_rate_24k", 24000),
            speech_token_rate=config_dict.get("speech_token_rate", 25),
            max_enc_cond_len_sec=config_dict.get("max_enc_cond_len_sec", 6.0),
            max_dec_cond_len_sec=config_dict.get("max_dec_cond_len_sec", 10.0),
        )

    @classmethod
    def from_json_file(cls, json_path: Union[str, Path]) -> "ChatterboxConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)