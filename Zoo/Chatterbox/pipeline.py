import random
from typing import Optional
import numpy as np
import torch
import torchaudio
import librosa
import torch.nn.functional as F

from Zoo.Chatterbox.utils.model_loader import load_chatterbox_mtl_v3
from Zoo.Chatterbox.SubModels.T3 import T3Cond

S3_SR = 16000
S3GEN_SR = 24000
S3_TOKEN_RATE = 25
ENC_COND_LEN = 6 * S3_SR       # 6 seconds maximum for T3 speech prompt
DEC_COND_LEN = 10 * S3GEN_SR   # 10 seconds maximum for S3Gen acoustic reference


def punc_norm(text: str) -> str:
    """Cleans up LLM punctuation, balances spaces, and ensures sentence closure."""
    if not text or len(text.strip()) == 0:
        return "You need to add some text for me to talk."

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
        (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)

    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


class ChatterboxPipeline:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = load_chatterbox_mtl_v3(device=self.device)
        self.ve = self.models["ve"]
        self.t3 = self.models["t3"]
        self.s3gen = self.models["s3gen"]
        self.tokenizer = self.models["tokenizer"]

    def generate_speech(
        self,
        text_prompt: str,
        reference_audio_path: str,
        language_id: str,
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        seed: Optional[int] = None,
    ) -> str:
        """Generate high-fidelity multilingual speech with zero-shot voice cloning."""
        if seed is not None and seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)

        # 1. Load reference audio at 24 kHz mono
        ref_24k, _ = librosa.load(reference_audio_path, sr=S3GEN_SR)
        ref_16k = librosa.resample(ref_24k, orig_sr=S3GEN_SR, target_sr=S3_SR)

        with torch.inference_mode():
            # 2. Extract S3Gen Reference Conditionals (CAMPPlus x-vector + 24k Mel)
            ref_dict = self.s3gen.embed_ref(ref_24k[:DEC_COND_LEN], S3GEN_SR, device=self.device)

            # 3. Extract T3 Reference Speech Prompt (discrete tokens at 25 Hz)
            plen = self.t3.hp.speech_cond_prompt_len  # 150 tokens = 6 seconds
            t3_prompt_tokens, _ = self.s3gen.tokenizer.forward([ref_16k[:ENC_COND_LEN]], max_len=plen)
            t3_prompt_tokens = torch.atleast_2d(t3_prompt_tokens).to(self.device)

            # 4. Extract VoiceEncoder 256-dim L2-normalized embedding for T3
            ve_emb = self.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR, as_spk=True)
            ve_emb_tensor = torch.from_numpy(ve_emb).to(device=self.device, dtype=torch.float32).unsqueeze(0)

            t3_cond = T3Cond(
                speaker_emb=ve_emb_tensor,
                cond_prompt_speech_tokens=t3_prompt_tokens,
                emotion_adv=torch.tensor([[[exaggeration]]], device=self.device, dtype=torch.float32),
            )

            # 5. Tokenize text prompt with language code prefix
            clean_text = punc_norm(text_prompt)
            text_tokens = self.tokenizer.text_to_tokens(clean_text, lang=language_id.lower()).to(self.device)

            # Add SOT (255) and EOT (0) tokens
            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            # 6. Autoregressive speech token inference via T3
            speech_tokens = self.t3.inference(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                temperature=temperature,
                top_p=1.0,
                min_p=0.05,
                repetition_penalty=1.2,
                cfg_weight=cfg_weight,
            )

            if speech_tokens.ndim >= 2:
                speech_tokens = speech_tokens[0]

            # Filter out non-speech/special IDs (valid vocabulary: < 6561)
            speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
            if speech_tokens.numel() == 0:
                raise RuntimeError("T3 produced no valid speech tokens.")

            # 7. S3Gen synthesis (Flow Matching + HiFT-Net Vocoder)
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=10,
            )

            # 8. Drop the final speech token's audio (degrades into boundary noise at EOS)
            # 25 Hz token rate -> 24000 // 25 = 960 samples per token
            n_tokens = int(speech_tokens.shape[-1])
            valid_samples = max(1, n_tokens - 1) * (S3GEN_SR // S3_TOKEN_RATE)
            wav = wav[:, :valid_samples].squeeze(0).cpu().float()

            torchaudio.save(output_path, wav.unsqueeze(0), S3GEN_SR)

        return output_path