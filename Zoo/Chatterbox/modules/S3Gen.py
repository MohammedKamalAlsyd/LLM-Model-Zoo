"""High-level S3Gen Synthesizer mapping discrete speech tokens to 24 kHz audio waveforms."""

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchaudio as ta

from Zoo.Chatterbox.configs import ChatterboxConfig
from Zoo.Chatterbox.modules.S3Flow import (
    CAMPPlus,
    CausalConditionalCFM,
    CausalMaskedDiffWithXvec,
    ConditionalDecoder,
    UpsampleConformerEncoder,
)
from Zoo.Chatterbox.modules.S3Tokenizer import S3Tokenizer, S3_SR
from Zoo.Chatterbox.modules.S3Vocoder import (
    ConvRNNF0Predictor,
    HiFTGenerator,
    mel_spectrogram,
)

S3GEN_SR: int = 24000


@lru_cache(100)
def get_resampler(src_sr: int, dst_sr: int, device: torch.device) -> ta.transforms.Resample:
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(nn.Module):
    """CFM acoustic decoder mapping discrete tokens to 80-bin mel-spectrograms."""

    def __init__(self, config: Optional[ChatterboxConfig] = None, meanflow: bool = False) -> None:
        super().__init__()
        self.config = config or ChatterboxConfig()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram
        self.speaker_encoder = CAMPPlus(memory_efficient=False)
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            input_size=self.config.s3_flow_config.input_size,
            output_size=self.config.s3_flow_config.input_size,
            attention_heads=self.config.s3_flow_config.conformer_heads,
            linear_units=self.config.s3_flow_config.conformer_linear_units,
            num_blocks=self.config.s3_flow_config.conformer_num_blocks,
            dropout_rate=self.config.s3_flow_config.conformer_dropout,
        )
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=self.config.s3_flow_config.output_size,
            causal=self.config.s3_flow_config.causal,
            channels=self.config.s3_flow_config.unet_channels,
            dropout=0.0,
            attention_head_dim=self.config.s3_flow_config.unet_attention_head_dim,
            n_blocks=self.config.s3_flow_config.unet_n_blocks,
            num_mid_blocks=self.config.s3_flow_config.unet_num_mid_blocks,
            num_heads=self.config.s3_flow_config.unet_num_heads,
            act_fn=self.config.s3_flow_config.act_fn,
            meanflow=self.meanflow,
        )
        decoder = CausalConditionalCFM(
            spk_emb_dim=self.config.s3_flow_config.output_size,
            estimator=estimator,
        )
        self.flow = CausalMaskedDiffWithXvec(
            input_size=self.config.s3_flow_config.input_size,
            output_size=self.config.s3_flow_config.output_size,
            spk_embed_dim=self.config.s3_flow_config.spk_embed_dim,
            vocab_size=self.config.s3_flow_config.vocab_size,
            encoder=encoder,
            decoder=decoder,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.flow.parameters()).dtype

    def embed_ref(
        self,
        ref_wav: Union[torch.Tensor, np.ndarray],
        ref_sr: int,
        device: Union[str, torch.device] = "auto",
    ) -> Dict[str, Any]:
        """Extracts CAMPPlus x-vectors, 24 kHz reference mels, and discrete speech prompt tokens."""
        target_device = self.device if device == "auto" else torch.device(device)
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()
        if ref_wav.device != target_device:
            ref_wav = ref_wav.to(target_device)
        if ref_wav.ndim == 1:
            ref_wav = ref_wav.unsqueeze(0)

        # 24 kHz mel spectrogram
        ref_wav_24 = ref_wav if ref_sr == S3GEN_SR else get_resampler(ref_sr, S3GEN_SR, target_device)(ref_wav)
        ref_wav_24 = ref_wav_24.to(device=target_device, dtype=self.dtype)
        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)

        # 16 kHz speaker x-vector and speech tokens
        ref_wav_16 = ref_wav if ref_sr == S3_SR else get_resampler(ref_sr, S3_SR, target_device)(ref_wav)
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        # Exact $2:1$ time frame alignment between mel frames and 25 Hz tokens
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, : ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return {
            "prompt_token": ref_speech_tokens.to(target_device),
            "prompt_token_len": ref_speech_token_lens,
            "prompt_feat": ref_mels_24,
            "prompt_feat_len": None,
            "embedding": ref_x_vector,
        }

    def forward(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: Dict[str, Any],
        n_cfm_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        speech_tokens = torch.atleast_2d(speech_tokens)
        speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)
        resolved_steps = n_cfm_timesteps or (2 if self.meanflow else self.config.s3_flow_config.default_cfm_steps)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=True,
            n_timesteps=resolved_steps,
            **ref_dict,
        )
        return output_mels


class S3Gen(S3Token2Mel):
    """Composite Discrete Token to 24 kHz Waveform Generator matching Chatterbox S3Token2Wav."""

    trim_fade: torch.Tensor

    def __init__(self, config: Optional[ChatterboxConfig] = None, meanflow: bool = False) -> None:
        super().__init__(config=config, meanflow=meanflow)

        self.mel2wav = HiFTGenerator(
            config=self.config.s3_vocoder_config,
            f0_predictor=ConvRNNF0Predictor(),
        )

        # 20ms cosine fade curve (480 samples at 24 kHz) preventing onset boundary clicks
        n_trim = S3GEN_SR // 50
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1.0) / 2.0
        self.register_buffer("trim_fade", trim_fade, persistent=False)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        # Dynamically inject persistent FFT window buffers if absent in state dictionary
        keys_to_ignore = ["tokenizer._mel_filters", "tokenizer.window"]
        for k in keys_to_ignore:
            if k not in state_dict and hasattr(self, "tokenizer"):
                state_dict[k] = getattr(self.tokenizer, k.split(".")[-1])
        return super().load_state_dict(state_dict, strict=strict)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: Dict[str, Any],
        n_cfm_timesteps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesizes high-fidelity 24 kHz waveform from discrete speech tokens.

        Args:
            speech_tokens: LongTensor of acoustic codes (vocab < 6561).
            ref_dict: Dictionary produced by `embed_ref`.
            n_cfm_timesteps: Number of Euler integration steps (default 10).

        Returns:
            wav: FloatTensor of shape (1, num_samples) in range [-0.99, 0.99].
            source: Harmonic excitation source tensor.
        """
        output_mels = self.forward(
            speech_tokens, ref_dict=ref_dict, n_cfm_timesteps=n_cfm_timesteps
        ).to(dtype=self.dtype)

        cache_source = torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype)
        output_wavs, output_sources = self.mel2wav.inference(speech_feat=output_mels, cache_source=cache_source)
        output_wavs[:, : len(self.trim_fade)] *= self.trim_fade
        return output_wavs, output_sources