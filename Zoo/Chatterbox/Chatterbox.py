"""Root Composite Model & End-to-End Synthesizer for Chatterbox V3 Multilingual TTS."""

import os
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple, Union
from huggingface_hub import snapshot_download
import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torchaudio

from Zoo.Chatterbox.configs import ChatterboxConfig
from Zoo.Chatterbox.modules.MTLTokenizer import MTLTokenizer
from Zoo.Chatterbox.modules.S3Gen import S3Gen
from Zoo.Chatterbox.modules.T3 import T3, T3Cond
from Zoo.Chatterbox.modules.VoiceEncoder import VoiceEncoder
from Zoo.Chatterbox.processing.ChatterboxProcessor import ChatterboxProcessor


class ChatterboxModel(nn.Module):
    """End-to-End Multilingual Zero-Shot Voice Cloning Architecture."""

    def __init__(self, config: Optional[ChatterboxConfig] = None) -> None:
        super().__init__()
        self.config = config or ChatterboxConfig()

        # 1. Dual Speaker Conditioning & Acoustic Models
        self.ve = VoiceEncoder(self.config.voice_encoder_config)
        self.t3 = T3(self.config.t3_config)
        self.s3gen = S3Gen(config=self.config, meanflow=False)

    @property
    def device(self) -> torch.device:
        return self.t3.device

    @property
    def dtype(self) -> torch.dtype:
        return self.t3.dtype

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "ResembleAI/chatterbox",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple["ChatterboxModel", ChatterboxProcessor]:
        """Downloads official weights and initializes Chatterbox with strict verification."""
        t3_ckpt = "t3_mtl23ls_v3.safetensors"
        print(f"Fetching weights for '{repo_id}' from Hugging Face Hub...")

        cache_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "ve.pt",
                    t3_ckpt,
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            )
        )

        model = cls()

        # 1. Load VoiceEncoder
        ve_state = torch.load(cache_dir / "ve.pt", map_location="cpu", weights_only=True)
        model.ve.load_state_dict(ve_state, strict=True)

        # 2. Load T3 Backbone
        t3_state: Dict[str, Any] = load_file(str(cache_dir / t3_ckpt))
        if "model" in t3_state and isinstance(t3_state["model"], list):
            t3_state = t3_state["model"][0]
        model.t3.load_state_dict(t3_state, strict=False)

        # 3. Load S3Gen
        s3gen_state = torch.load(cache_dir / "s3gen.pt", map_location="cpu", weights_only=True)
        model.s3gen.load_state_dict(s3gen_state, strict=True)

        # 4. Initialize Processor
        vocab_file = cache_dir / "grapheme_mtl_merged_expanded_v1.json"
        cj_file = cache_dir / "Cangjie5_TC.json"
        tokenizer = MTLTokenizer(vocab_path=vocab_file, cj_path=cj_file)
        processor = ChatterboxProcessor(tokenizer=tokenizer, config=model.config)

        model.to(device=device)
        if dtype is not None:
            model.to(dtype=dtype)

        model.eval()
        print(f"✓ Chatterbox successfully initialized on {device.upper()}.")
        return model, processor

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: torch.Tensor,
        ref_24k: np.ndarray,
        ref_16k: np.ndarray,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        n_cfm_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Synthesizes raw audio waveform from preprocessed tokens and reference audio arrays.

        Args:
            text_tokens: LongTensor of shape (1, seq_len).
            ref_24k: Float32 audio array at 24000 Hz.
            ref_16k: Float32 audio array at 16000 Hz.
            exaggeration: Emotion intensity scalar.
            cfg_weight: Classifier-Free Guidance extrapolation multiplier.
            temperature: Softmax sampling temperature.
            n_cfm_timesteps: Number of Euler integration steps.
            seed: Optional integer seed for deterministic reproduction.

        Returns:
            FloatTensor waveform of shape (1, num_samples) at 24000 Hz.
        """
        if seed is not None and seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)

        target_dev = self.device
        sr_16k = self.config.sample_rate_16k
        sr_24k = self.config.sample_rate_24k
        enc_cond_len = int(self.config.max_enc_cond_len_sec * sr_16k)
        dec_cond_len = int(self.config.max_dec_cond_len_sec * sr_24k)

        # Stage 1: Extract S3Gen Reference Conditionals
        ref_dict = self.s3gen.embed_ref(ref_24k[:dec_cond_len], sr_24k, device=target_dev)

        # Stage 2: Extract T3 Reference Speech Prompt (25 Hz codes)
        prompt_len = self.config.t3_config.speech_cond_prompt_len
        t3_prompt_tokens, _ = self.s3gen.tokenizer.forward([ref_16k[:enc_cond_len]], max_len=prompt_len)
        t3_prompt_tokens = torch.atleast_2d(t3_prompt_tokens).to(target_dev)

        # Stage 3: Extract VoiceEncoder 256-dim L2-normalized embedding
        ve_emb = self.ve.embeds_from_wavs([ref_16k], sample_rate=sr_16k, as_spk=True)
        ve_emb_tensor = torch.from_numpy(ve_emb).to(device=target_dev, dtype=torch.float32).unsqueeze(0)

        # Stage 4: Construct T3 Conditioning Container
        t3_cond = T3Cond(
            speaker_emb=ve_emb_tensor,
            cond_prompt_speech_tokens=t3_prompt_tokens,
            emotion_adv=torch.tensor([[[exaggeration]]], device=target_dev, dtype=torch.float32),
        )

        # Stage 5: Autoregressive Speech Token Inference
        speech_tokens = self.t3.inference(
            t3_cond=t3_cond,
            text_tokens=text_tokens.to(target_dev),
            temperature=temperature,
            top_p=1.0,
            min_p=0.05,
            repetition_penalty=1.2,
            cfg_weight=cfg_weight,
        )

        if speech_tokens.ndim >= 2:
            speech_tokens = speech_tokens[0]

        # Filter out non-speech and special IDs (valid speech vocab: < 6561)
        speech_tokens = speech_tokens[speech_tokens < 6561].to(target_dev)
        if speech_tokens.numel() == 0:
            raise RuntimeError("T3 produced zero valid speech tokens.")

        # Stage 6: Acoustic Flow Matching + Neural Source Filter Vocoder
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
        )

        # Drop the final speech token's samples to prevent boundary click artifacts
        token_rate = self.config.speech_token_rate
        n_tokens = int(speech_tokens.shape[-1])
        valid_samples = max(1, n_tokens - 1) * (sr_24k // token_rate)
        wav = wav[:, :valid_samples].squeeze(0).cpu().float()

        return wav.unsqueeze(0)