from functools import lru_cache
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchaudio as ta

from Zoo.Chatterbox.SubModels.S3Tokenizer import S3Tokenizer, S3_SR
from Zoo.Chatterbox.SubModels.S3Flow import (
    CAMPPlus,
    UpsampleConformerEncoder,
    ConditionalDecoder,
    CausalConditionalCFM,
    CausalMaskedDiffWithXvec,
)
from Zoo.Chatterbox.SubModels.S3Vocoder import (
    mel_spectrogram,
    HiFTGenerator,
    ConvRNNF0Predictor,
)

S3GEN_SR = 24000


@lru_cache(100)
def get_resampler(src_sr: int, dst_sr: int, device: torch.device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(nn.Module):
    """CFM acoustic decoder mapping discrete tokens -> 80-bin mel-spectrograms."""
    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram
        self.speaker_encoder = CAMPPlus(memory_efficient=False)
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            input_size=512, output_size=512, attention_heads=8,
            linear_units=2048, num_blocks=6, dropout_rate=0.1
        )
        estimator = ConditionalDecoder(
            in_channels=320, out_channels=80, causal=True, channels=[256],
            dropout=0.0, attention_head_dim=64, n_blocks=4, num_mid_blocks=12,
            num_heads=8, act_fn="gelu", meanflow=self.meanflow
        )
        decoder = CausalConditionalCFM(spk_emb_dim=80, estimator=estimator)
        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.flow.parameters()).dtype

    def embed_ref(self, ref_wav: Union[torch.Tensor, np.ndarray], ref_sr: int, device="auto") -> dict:
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()
        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)
        if ref_wav.ndim == 1:
            ref_wav = ref_wav.unsqueeze(0)

        # 24 kHz reference mel
        ref_wav_24 = ref_wav if ref_sr == S3GEN_SR else get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
        ref_wav_24 = ref_wav_24.to(device=device, dtype=self.dtype)
        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)

        # 16 kHz reference speaker x-vector and speech tokens
        ref_wav_16 = ref_wav if ref_sr == S3_SR else get_resampler(ref_sr, S3_SR, device)(ref_wav)
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        # Alignment
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, : ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return {
            "prompt_token": ref_speech_tokens.to(device),
            "prompt_token_len": ref_speech_token_lens,
            "prompt_feat": ref_mels_24,
            "prompt_feat_len": None,
            "embedding": ref_x_vector,
        }

    def forward(self, speech_tokens: torch.Tensor, ref_dict: dict, n_cfm_timesteps: Optional[int] = None) -> torch.Tensor:
        speech_tokens = torch.atleast_2d(speech_tokens)
        speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)
        resolved_n_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=True,
            n_timesteps=resolved_n_timesteps,
            **ref_dict,
        )
        return output_mels


class S3Gen(S3Token2Mel):
    """End-to-end discrete token to 24 kHz waveform generator matching Chatterbox S3Token2Wav."""
    trim_fade: torch.Tensor
    def __init__(self, meanflow: bool = False):
        super().__init__(meanflow=meanflow)

        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=ConvRNNF0Predictor(),
        )

        n_trim = S3GEN_SR // 50  # 20ms = 480 samples
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Allow missing persistent buffers in tokenizer without throwing when strict=True
        keys_to_ignore = ["tokenizer._mel_filters", "tokenizer.window"]
        for k in keys_to_ignore:
            if k not in state_dict and hasattr(self, "tokenizer"):
                state_dict[k] = getattr(self.tokenizer, k.split(".")[-1])
        return super().load_state_dict(state_dict, strict=strict)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: dict,
        n_cfm_timesteps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_mels = self.forward(speech_tokens, ref_dict=ref_dict, n_cfm_timesteps=n_cfm_timesteps).to(dtype=self.dtype)
        cache_source = torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype)
        output_wavs, output_sources = self.mel2wav.inference(speech_feat=output_mels, cache_source=cache_source)
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade
        return output_wavs, output_sources