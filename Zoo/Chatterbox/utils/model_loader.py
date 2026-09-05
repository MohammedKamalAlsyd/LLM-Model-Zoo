import os
from typing import Dict, Any, cast
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from Zoo.Chatterbox.SubModels.T3 import T3, T3Config
from Zoo.Chatterbox.SubModels.S3Gen import S3Gen
from Zoo.Chatterbox.SubModels.voice_encoder import VoiceEncoder


def load_chatterbox_turbo(device: str = "cuda") -> Dict[str, Any]:
    repo_id = "ResembleAI/chatterbox-turbo"
    print(f"Fetching weights from {repo_id} (using HF Cache)...")
    
    cache_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.txt"],
        token=os.getenv("HF_TOKEN")
    )
    
    print("Initializing Clean Architectures...")
    
    # 1. Voice Encoder
    ve = VoiceEncoder().to(device)
    ve_state = load_file(os.path.join(cache_dir, "ve.safetensors"))
    ve.load_state_dict(ve_state, strict=True)
    ve.eval()

    # 2. T3 (Turbo Config)
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    
    t3 = T3(hp).to(device)
    t3_state_raw = load_file(os.path.join(cache_dir, "t3_turbo_v1.safetensors"))
    
    # Safely cast the state dict to satisfy Pylance
    if "model" in t3_state_raw and isinstance(t3_state_raw["model"], list):
        t3_state = cast(Dict[str, torch.Tensor], t3_state_raw["model"][0])
    else:
        t3_state = cast(Dict[str, torch.Tensor], t3_state_raw)
    
    # Remove unused HuggingFace GPT2 text embedding (we use our own text_emb)
    t3_state.pop("tfmr.wte.weight", None) 
    t3.load_state_dict(t3_state, strict=True)
    
    if hasattr(t3.tfmr, 'wte'):
        del t3.tfmr.wte 
    t3.eval()

    # 3. S3Gen (Flow + Vocoder)
    s3gen = S3Gen(meanflow=True).to(device)
    s3gen_mega_state = load_file(os.path.join(cache_dir, "s3gen_meanflow.safetensors"))
    
    # Filter out generated tokenizer buffers to ensure strict=True passes cleanly
    clean_s3gen_state = {
        k: v for k, v in s3gen_mega_state.items() 
        if not k.startswith("tokenizer.") and not k.startswith("trim_fade")
    }
    
    s3gen.load_state_dict(clean_s3gen_state, strict=True)
    s3gen.eval()

    # 4. Text Tokenizer (Turbo uses standard AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ All models successfully loaded with strict=True!")
    
    return {
        "ve": ve,
        "t3": t3,
        "s3gen": s3gen,
        "tokenizer": tokenizer,
        "device": device
    }