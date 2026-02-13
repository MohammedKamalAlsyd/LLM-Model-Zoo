import os
import sys
import json
import torch
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Add Zoo to path to allow imports from your structure
sys.path.append(os.path.join(os.path.dirname(__file__), "Zoo", "PaliGemma2"))

# Import your custom modules
from PaliGemma2 import PaliGemma2Config, PaliGemma2ForConditionalGeneration
from utils.PaliGemma2Processor import PaliGemma2Processor
from utils.KVCache import KVCache

# --- Configuration ---
# Replace this with the Hugging Face Repo ID that matches your architecture.
# Since this architecture mixes Gemma2 and SigLip, ensure the weights match.
# If this is a custom model you trained, put your repo ID here.
HF_MODEL_ID = "google/paligemma-3b-pt-224" # Placeholder: Adjust if using specific Gemma2 weights
LOCAL_SAVE_DIR = "./saved_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

def load_config():
    config_path = os.path.join("Zoo", "PaliGemma2", "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # We need to construct the config object manually based on the dict
    # or rely on the __post_init__ defaults in your class if strict mapping isn't needed.
    # Here we instantiate a blank config and let the class defaults/JSON structure handle it.
    
    # Create sub-configs based on the dictionary structure
    from SubModels.Gemma2 import Gemma2Config
    from SubModels.SigLip import SigLipVisionConfig
    
    text_cfg_data = config_dict.get("text_config", {})
    vision_cfg_data = config_dict.get("vision_config", {})

    text_config = Gemma2Config(
        vocab_size=text_cfg_data.get("vocab_size", 257216),
        hidden_size=text_cfg_data.get("hidden_size", 2304),
        intermediate_size=text_cfg_data.get("intermediate_size", 9216),
        num_attention_heads=text_cfg_data.get("num_attention_heads", 8),
        num_hidden_layers=text_cfg_data.get("num_hidden_layers", 26),
        num_key_value_heads=text_cfg_data.get("num_key_value_heads", 4),
        sliding_window=text_cfg_data.get("sliding_window", 4096),
        head_dim=text_cfg_data.get("head_dim", 2304 // 8),
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        pad_token_id=config_dict.get("pad_token_id", 0),
        eos_token_id=config_dict.get("eos_token_id", 1),
        bos_token_id=config_dict.get("bos_token_id", 2),
        image_token_id=config_dict.get("image_token_index", 257152),
        attention_dropout=0.0,
        rope_theta=10000, # Default for Gemma
        query_pre_attn_scalar=text_cfg_data.get("query_pre_attn_scalar", 256),
        attn_logit_softcapping=text_cfg_data.get("attn_logit_softcapping", 50.0),
        final_logit_softcapping=text_cfg_data.get("final_logit_softcapping", 30.0)
    )

    vision_config = SigLipVisionConfig(
        hidden_size=vision_cfg_data.get("hidden_size", 1152),
        intermediate_size=vision_cfg_data.get("intermediate_size", 4304),
        num_attention_heads=vision_cfg_data.get("num_attention_heads", 16),
        num_hidden_layers=vision_cfg_data.get("num_hidden_layers", 27),
        patch_size=vision_cfg_data.get("patch_size", 14),
        projection_dim=vision_cfg_data.get("projection_dim", 2304),
        image_size=224, # Fixed based on SigLip default
        num_channels=3
    )

    config = PaliGemma2Config(
        text_config=text_config,
        vision_config=vision_config,
        projection_dim=config_dict.get("projection_dim", 2304),
        image_token_index=config_dict.get("image_token_index", 257152),
        torch_dtype="bfloat16"
    )
    
    return config

def download_and_load_weights(model, hf_id, local_dir):
    """
    Downloads weights from HF, loads them, and maps keys to your custom architecture.
    Saves the aligned weights locally.
    """
    weights_path = os.path.join(local_dir, "paligemma2_custom.pth")

    if os.path.exists(weights_path):
        print(f"Loading weights from local cache: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        print(f"Downloading weights from Hugging Face: {hf_id}...")
        try:
            model_path = snapshot_download(repo_id=hf_id, allow_patterns=["*.safetensors", "*.bin"])
            
            # Load safetensors preferably
            safetensors_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_file):
                state_dict = load_file(safetensors_file)
            else:
                bin_file = os.path.join(model_path, "pytorch_model.bin")
                state_dict = torch.load(bin_file, map_location="cpu")
            
            print("Weights downloaded. Remapping keys if necessary...")
            
            # --- Key Mapping Logic ---
            # Your model uses: language_model, vision_tower, multi_modal_projector
            # Official HF weights might differ slightly.
            # This is a heuristic mapping.
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Remap common mismatches here if your architecture differs from official keys
                # Example: "text_model" -> "language_model"
                if key.startswith("text_model."):
                    new_key = key.replace("text_model.", "language_model.")
                
                new_state_dict[new_key] = value

            state_dict = new_state_dict

            # Ensure directory exists
            os.makedirs(local_dir, exist_ok=True)
            print(f"Saving processed weights to {weights_path}...")
            torch.save(state_dict, weights_path)
            
        except Exception as e:
            print(f"Error downloading/processing weights: {e}")
            print("Initializing with random weights for demonstration purposes.")
            return model

    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {len(missing)}")
    if unexpected:
        print(f"Warning: Unexpected keys: {len(unexpected)}")
        
    return model

# --- Inference Logic ---

@torch.no_grad()
def generate_text(
    model: PaliGemma2ForConditionalGeneration,
    processor: PaliGemma2Processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
):
    model.eval()
    
    # 1. Process Inputs
    # Note: PaliGemma prompts usually do not need a system prompt, just the query
    inputs = processor(text=prompt, image=image, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE).to(DTYPE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    # 2. Setup KV Cache
    kv_cache = KVCache()
    
    generated_tokens = []
    
    # 3. Generation Loop
    # We only have a forward() method, so we implement greedy/sample decoding manually
    
    curr_input_ids = input_ids
    
    for _ in range(max_new_tokens):
        # Forward pass
        # pixel_values must always be passed due to the structure of your model's forward
        # logic, even though they are only mathematically relevant in the first step/prefill.
        outputs = model(
            input_ids=curr_input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        
        logits = outputs["logits"] # (B, Seq, Vocab)
        next_token_logits = logits[:, -1, :]
        
        # Apply Temperature
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        # Check EOS
        if next_token.item() == model.config.text_config.eos_token_id:
            break
            
        generated_tokens.append(next_token.item())
        
        # Prepare for next iteration
        curr_input_ids = next_token # Pass only the new token
        
        # Update attention mask for the new token (simple logic: append 1)
        # Note: Your model's _merge logic handles position IDs based on cache presence
        # so explicit mask expansion logic might be handled internally or via the mask passed.
        # For simple generation, we often just pass the new token ID.
        attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=DEVICE, dtype=attention_mask.dtype)], dim=1)

    # 4. Decode Output
    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text

# --- Main Application ---

def main():
    print("Initializing PaliGemma2 Service...")
    
    # 1. Load Configuration
    config = load_config()
    
    # 2. Initialize Model
    print("Building Model Architecture...")
    model = PaliGemma2ForConditionalGeneration(config)
    
    # 3. Load Weights (Download -> Save -> Load)
    model = download_and_load_weights(model, HF_MODEL_ID, LOCAL_SAVE_DIR)
    
    model.to(DEVICE).to(DTYPE)
    model.eval()
    
    # 4. Initialize Processor
    # Using standard Gemma tokenizer as base
    print("Loading Tokenizer...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    except:
        print("Fallback to generic gemma tokenizer")
        base_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        
    processor = PaliGemma2Processor(
        tokenizer=base_tokenizer,
        image_tokens=config.vision_config.patch_size, # From config (usually 256)
        image_size=config.vision_config.image_size
    )
    
    # 5. Gradio Interface
    def predict(image, text, max_tokens, temp):
        if image is None:
            return "Please upload an image."
        if not text:
            text = "describe this image" # default prompt
            
        try:
            result = generate_text(
                model=model,
                processor=processor,
                image=image,
                prompt=text,
                max_new_tokens=int(max_tokens),
                temperature=float(temp)
            )
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating text: {str(e)}"

    print("Starting Gradio Server...")
    
    with gr.Blocks(title="PaliGemma2 Model Zoo") as demo:
        gr.Markdown("# PaliGemma2 Multimodal Inference")
        gr.Markdown(f"Running on **{DEVICE}** with dtype **{DTYPE}**")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                input_text = gr.Textbox(label="Prompt", placeholder="Describe this image...", lines=2)
                
                with gr.Accordion("Advanced Settings", open=False):
                    slider_tokens = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max New Tokens")
                    slider_temp = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                
                submit_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Model Output", interactive=False)
        
        submit_btn.click(
            fn=predict,
            inputs=[input_image, input_text, slider_tokens, slider_temp],
            outputs=[output_text]
        )

    demo.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()