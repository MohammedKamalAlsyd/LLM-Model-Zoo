import os
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
        cfg_weight: float = 0.5
    ):
        """Generates TTS using zero-shot cloning."""
        
        # 1. Load Reference Audio
        ref_wav, ref_sr = torchaudio.load(reference_audio_path)
        ref_wav = ref_wav.mean(dim=0) # mono

        # 2. Extract Acoustic Conditioning
        with torch.inference_mode():
            # Speaker Vector (Voice Encoder)
            ve_emb = self.ve.extract_speaker_embedding(ref_wav).unsqueeze(0)
            
            # S3Gen Voice Prompt Dict (Mel + Prompt Tokens + Spk X-Vector)
            ref_dict = self.s3gen.embed_ref(ref_wav, ref_sr)
            
            # T3 Perceiver Prompt (Variable length audio mapped to 32 tokens)
            # Take up to 150 tokens (6 seconds) for the perceiver prompt
            prompt_tokens_for_t3 = ref_dict["prompt_tokens"][:, :150]
            cond_prompt_emb = self.s3gen.tokenizer.input_embedding(prompt_tokens_for_t3)
            
            # Assemble T3 Conditionals
            t3_cond = T3Cond(
                speaker_emb=ve_emb.to(self.device),
                cond_prompt_speech_emb=cond_prompt_emb.to(self.device),
                emotion_adv=torch.tensor([[[exaggeration]]], device=self.device)
            )

            # 3. Process Text
            text_tokens = self.tokenizer.text_to_tokens(text_prompt, lang=language_id).to(self.device)
            
            # Pad with SOT and EOT tokens
            sot, eot = self.t3.hp.start_text_token, self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            # 4. Generate S3 Tokens via T3 (Auto-handles CFG natively inside `generate`)
            speech_tokens = self.t3.generate(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                temperature=0.8,
                top_p=1.0,
                min_p=0.05,
                repetition_penalty=1.2,
                cfg_weight=cfg_weight
            )
            
            # Stop token pruning
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)

            # 5. Decode to Audio via S3Gen
            wav = self.s3gen.generate(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=10 # Default for non-meanflow standard models
            )
            
            # Save Output
            wav = wav.squeeze(0).cpu()
            torchaudio.save(output_path, wav.unsqueeze(0), 24000)
            
        return output_path