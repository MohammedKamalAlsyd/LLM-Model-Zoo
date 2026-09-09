import torch
import torchaudio
import torch.nn.functional as F

from Zoo.Chatterbox.utils.model_loader import load_chatterbox_mtl_v3
from Zoo.Chatterbox.SubModels.T3 import T3Cond

class ChatterboxPipeline:
    def __init__(self, device="cuda"):
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
    ):
        """Generate TTS using zero-shot voice cloning."""

        # 1. Load reference audio and convert it to mono.
        ref_wav, ref_sr = torchaudio.load(reference_audio_path)
        if ref_wav.size(0) > 1:
            ref_wav = ref_wav.mean(dim=0, keepdim=True)
        ref_wav = ref_wav.squeeze(0)

        with torch.inference_mode():
            # 2. T3 speaker conditioning: VoiceEncoder expects 16 kHz audio.
            if ref_sr != 16000:
                ref_wav_16k = torchaudio.functional.resample(
                    ref_wav, ref_sr, 16000
                )
            else:
                ref_wav_16k = ref_wav

            ve_emb = self.ve.extract_speaker_embedding(ref_wav_16k).unsqueeze(0)

            # 3. S3Gen reference conditioning.
            ref_dict = self.s3gen.embed_ref(ref_wav, ref_sr)

            # 4. T3 reference speech prompt.
            prompt_token_for_t3 = ref_dict["prompt_token"][:, :150].to(self.device)

            cond_prompt_emb = self.t3.speech_emb(prompt_token_for_t3)
            cond_prompt_emb = (
                cond_prompt_emb
                + self.t3.speech_pos_emb(prompt_token_for_t3)
            )

            t3_cond = T3Cond(
                speaker_emb=ve_emb.to(self.device),
                cond_prompt_speech_emb=cond_prompt_emb.to(self.device),
                emotion_adv=torch.tensor(
                    [[[exaggeration]]],
                    device=self.device,
                    dtype=ve_emb.dtype,
                ),
            )

            # 5. Tokenize text.
            text_tokens = self.tokenizer.text_to_tokens(
                text_prompt,
                lang=language_id,
            ).to(self.device)

            # Add T3 start/end text tokens.
            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token

            text_tokens = F.pad(
                text_tokens,
                (1, 0),
                value=sot,
            )
            text_tokens = F.pad(
                text_tokens,
                (0, 1),
                value=eot,
            )

            # 6. T3: text -> S3 speech tokens.
            speech_tokens = self.t3.generate(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                temperature=0.8,
                top_p=1.0,
                min_p=0.05,
                repetition_penalty=1.2,
                cfg_weight=cfg_weight,
            )

            # Batch size = 1.
            if speech_tokens.ndim >= 2:
                speech_tokens = speech_tokens[0]

            # Keep only valid S3 speech-token IDs.
            speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)

            if speech_tokens.numel() == 0:
                raise RuntimeError(
                    "T3 produced no valid S3 speech tokens."
                )

            # 7. S3Gen: speech tokens + reference conditioning -> waveform.
            wav = self.s3gen.generate(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=10,
            )

            # 8. Remove final token's audio.
            # S3 tokens are 25 Hz => 960 samples/token at 24 kHz.
            n_tokens = int(speech_tokens.shape[-1])
            valid_tokens = max(1, n_tokens - 1)
            samples_per_token = 24000 // 25

            max_samples = valid_tokens * samples_per_token
            wav = wav[..., :max_samples]

            # Do not apply additional peak normalization.
            wav = wav.detach().cpu().squeeze()

            if wav.ndim != 1:
                raise RuntimeError(
                    f"Unexpected waveform shape: {tuple(wav.shape)}"
                )

            torchaudio.save(
                output_path,
                wav.unsqueeze(0),
                24000,
            )

        return output_path