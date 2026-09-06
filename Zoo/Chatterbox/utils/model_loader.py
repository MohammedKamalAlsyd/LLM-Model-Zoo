import os
from typing import Dict, cast
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Fixed imports based on your structure
from Zoo.Chatterbox.SubModels.T3 import T3, T3Config
from Zoo.Chatterbox.SubModels.S3Gen import S3Gen
from Zoo.Chatterbox.SubModels.voice_encoder import VoiceEncoder
from Zoo.Chatterbox.SubModels.MTLTokenizer import MTLTokenizer

def load_chatterbox_mtl_v3(device: str = "cuda"):
    repo_id = "ResembleAI/chatterbox"
    t3_ckpt = "t3_mtl23ls_v3.safetensors"
    
    print(f"Fetching V3 Multilingual weights from {repo_id}...")
    
    cache_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["ve.pt", t3_ckpt, "s3gen.pt", "*.json"],
        token=os.getenv("HF_TOKEN")
    )
    
    print("Initializing Clean Architectures...")
    
    # 1. Voice Encoder
    ve = VoiceEncoder().to(device)
    ve_state = torch.load(os.path.join(cache_dir, "ve.pt"), map_location="cpu", weights_only=True)
    ve.load_state_dict(ve_state, strict=True)
    ve.eval()

    # 2. T3 (Multilingual Config)
    hp = T3Config(
        text_tokens_dict_size=2454,  # Multilingual vocab size
        llama_config_name="Llama_520M",
        speech_tokens_dict_size=8194,
        input_pos_emb="learned",
        speaker_embed_size=256,
        emotion_adv=True
    )
    t3 = T3(hp).to(device)
    
    # Load and explicitly cast to Dict[str, Tensor] so Pylance stops complaining
    t3_state_raw = load_file(os.path.join(cache_dir, t3_ckpt))
    t3_state = cast(Dict[str, torch.Tensor], t3_state_raw)
    
    # Clean T3 state for strict loading
    t3.load_state_dict(t3_state, strict=True)
    t3.eval()

    # 3. S3Gen (Vocoder + Flow)
    s3gen = S3Gen(meanflow=False).to(device)  # MTL uses standard Flow Matching
    s3gen_state = torch.load(os.path.join(cache_dir, "s3gen.pt"), map_location="cpu", weights_only=True)
    s3gen.load_state_dict(s3gen_state, strict=True)
    s3gen.eval()

    # 4. Multilingual Tokenizer
    vocab_path = os.path.join(cache_dir, "grapheme_mtl_merged_expanded_v1.json")
    cj_path = os.path.join(cache_dir, "Cangjie5_TC.json")
    tokenizer = MTLTokenizer(vocab_path=vocab_path, cj_path=cj_path)

    print("✅ V3 Multilingual models successfully loaded with strict=True!")
    
    return {
        "ve": ve,
        "t3": t3,
        "s3gen": s3gen,
        "tokenizer": tokenizer,
        "device": device
    }