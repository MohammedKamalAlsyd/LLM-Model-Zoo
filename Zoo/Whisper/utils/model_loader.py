import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from Zoo.Whisper.SubModels.Encoder import WhisperEncoder
from Zoo.Whisper.SubModels.Decoder import WhisperDecoder

def load_whisper_weights(device: str = "cpu"):
    print("Downloading/Loading Whisper large-v3-turbo weights...")
    repo_id = "openai/whisper-large-v3-turbo"
    
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    hf_state_dict = load_file(weights_path)
    
    print("Initializing from-scratch architectures...")
    encoder = WhisperEncoder().to(device)
    decoder = WhisperDecoder().to(device)
    
    encoder_dict = {}
    decoder_dict = {}

    print("Routing and mapping weights to custom layers...")
    for key, value in hf_state_dict.items():
        # -------------------------------------------------------------
        # 1. Route to Encoder
        # -------------------------------------------------------------
        if key.startswith("model.encoder."):
            new_key = key.replace("model.encoder.", "")
            
            # Map specific layer names to our clean architecture
            new_key = new_key.replace("self_attn.q_proj.", "self_attn.q_proj.")
            new_key = new_key.replace("self_attn.k_proj.", "self_attn.k_proj.")
            new_key = new_key.replace("self_attn.v_proj.", "self_attn.v_proj.")
            new_key = new_key.replace("self_attn.out_proj.", "self_attn.out_proj.")
            
            encoder_dict[new_key] = value

        # -------------------------------------------------------------
        # 2. Route to Decoder
        # -------------------------------------------------------------
        elif key.startswith("model.decoder."):
            new_key = key.replace("model.decoder.", "")
            
            # Map Decoder Attention
            new_key = new_key.replace("self_attn.q_proj.", "self_attn.q_proj.")
            new_key = new_key.replace("encoder_attn.q_proj.", "encoder_attn.q_proj.")
            
            decoder_dict[new_key] = value

    print("Loading mapped weights into modules...")
    encoder.load_state_dict(encoder_dict, strict=True)
    decoder.load_state_dict(decoder_dict, strict=True)
    
    print("Models successfully loaded!")
    
    return encoder, decoder