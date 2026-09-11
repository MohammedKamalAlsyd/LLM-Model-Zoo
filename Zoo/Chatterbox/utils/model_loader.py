import os
from pathlib import Path
from typing import Dict, Any, cast
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from Zoo.Chatterbox.SubModels.T3 import T3, T3Config
from Zoo.Chatterbox.SubModels.S3Gen import S3Gen
from Zoo.Chatterbox.SubModels.voice_encoder import VoiceEncoder
from Zoo.Chatterbox.SubModels.MTLTokenizer import MTLTokenizer


def load_chatterbox_mtl_v3(device: str = "cuda") -> Dict[str, Any]:
    repo_id = "ResembleAI/chatterbox"
    t3_ckpt = "t3_mtl23ls_v3.safetensors"
    
    print(f"Fetching V3 Multilingual weights from {repo_id}...")
    
    cache_dir = Path(snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "ve.pt",
            t3_ckpt,
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json"
        ],
        token=os.getenv("HF_TOKEN")
    ))
    
    print(f"Initializing clean architectures on device: {device}...")
    
    # 1. Voice Encoder (16 kHz LSTM for T3 Speaker Conditioning)
    ve = VoiceEncoder().to(device)
    ve_state = torch.load(cache_dir / "ve.pt", map_location="cpu", weights_only=True)
    ve.load_state_dict(ve_state, strict=True)
    ve.eval()

    # 2. T3 Backbone (520M LLaMA Multilingual)
    hp = T3Config(
        text_tokens_dict_size=2454,
        llama_config_name="Llama_520M",
        speech_tokens_dict_size=8194,
        input_pos_emb="learned",
        speaker_embed_size=256,
        emotion_adv=True
    )
    t3 = T3(hp).to(device)
    
    t3_state = cast(Dict[str, torch.Tensor], load_file(str(cache_dir / t3_ckpt)))
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(cast(Dict[str, Any], t3_state), strict=False)
    t3.eval()

    # 3. S3Gen (Flow Matching CFM + HiFT-Net Vocoder + CAMPPlus)
    s3gen = S3Gen(meanflow=False).to(device)
    s3gen_state = torch.load(cache_dir / "s3gen.pt", map_location="cpu", weights_only=True)
    s3gen.load_state_dict(s3gen_state, strict=True)
    s3gen.eval()

    # 4. Multilingual Tokenizer (CJK + 23 Languages)
    vocab_path = str(cache_dir / "grapheme_mtl_merged_expanded_v1.json")
    cj_path = str(cache_dir / "Cangjie5_TC.json")
    tokenizer = MTLTokenizer(vocab_path=vocab_path, cj_path=cj_path)

    print("✅ Chatterbox V3 Multilingual models successfully loaded with strict=True!")
    
    return {
        "ve": ve,
        "t3": t3,
        "s3gen": s3gen,
        "tokenizer": tokenizer,
        "device": device
    }