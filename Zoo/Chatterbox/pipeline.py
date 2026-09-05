import os
import torch
import torchaudio
from typing import Optional

from Zoo.Chatterbox.utils.model_loader import load_chatterbox_turbo
from Zoo.Chatterbox.SubModels.T3 import T3Cond

def punc_norm(text: str) -> str:
    """Standardizes punctuation for LLM inputs."""
    if len(text) == 0:
        return "You need to add some text."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    replacements = [
        ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), 
        (" ,", ","), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text


class ChatterboxPipeline:
    """Unified pipeline orchestrating VoiceEncoder, T3, and S3Gen."""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Chatterbox components on {self.device.upper()}...")
        
        models = load_chatterbox_turbo(device=self.device)
        self.ve = models["ve"]
        self.t3 = models["t3"]
        self.s3gen = models["s3gen"]
        self.tokenizer = models["tokenizer"]
        self.sample_rate = 24000

    @torch.inference_mode()
    def generate_speech(
        self,
        text_prompt: str,
        reference_audio_path: str,
        output_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        """
        Generates zero-shot voice cloned speech and saves it to output_path.
        """
        # 1. Load and prepare reference audio
        ref_wav, ref_sr = torchaudio.load(reference_audio_path)
        ref_wav = ref_wav.mean(dim=0)  # Mono conversion [T]
        
        # 2. Extract Acoustic & Timbre features
        ve_emb = self.ve.extract_speaker_embedding(ref_wav.numpy()).unsqueeze(0)
        ref_dict = self.s3gen.embed_ref(ref_wav, ref_sr)
        
        t3_cond = T3Cond(
            speaker_emb=ve_emb.to(self.device),
            cond_prompt_speech_emb=None,
            emotion_adv=None
        )

        # 3. Tokenize text
        clean_text = punc_norm(text_prompt)
        text_inputs = self.tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_inputs.input_ids.to(self.device)

        # 4. Generate discrete S3 speech tokens with T3
        speech_tokens = self.t3.generate(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=1500,
            cfg_weight=0.0  # Turbo architecture uses CFG=0
        )

        # 5. Clean up tokens and append trailing silence (4299)
        speech_tokens = speech_tokens[speech_tokens < 6561]
        silence = torch.tensor([4299, 4299, 4299], dtype=torch.long, device=self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        # 6. Flow Match & Vocode to raw 24kHz audio
        wav = self.s3gen.generate(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=2  # 2-step fast distillation
        )

        # 7. Save output audio
        wav = wav.detach().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torchaudio.save(output_path, wav, sample_rate=self.sample_rate)
        
        return output_path