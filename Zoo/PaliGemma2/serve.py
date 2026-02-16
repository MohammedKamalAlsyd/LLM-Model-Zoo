import os
import sys
import torch
import gradio as gr
from PIL import Image
from pathlib import Path
from huggingface_hub import snapshot_download, login
from safetensors import safe_open
import gc
from dotenv import load_dotenv

# Environment Setup
load_dotenv()
# Ensure we can find the Zoo folder
sys.path.append(os.path.join(os.path.dirname(__file__), "Zoo", "PaliGemma2"))

# Import Model
from PaliGemma2 import PaliGemma2Config, PaliGemma2ForConditionalGeneration
from utils.PaliGemma2Processor import PaliGemma2Processor
from utils.KVCache import KVCache
from transformers import AutoTokenizer

# --- Constants ---
HF_REPO = "google/paligemma2-3b-pt-224"
LOCAL_DIR = "./saved_model/Paligemma2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 if supported (Amperere+ GPUs), otherwise float32
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

# Authentication
if token := os.getenv("HUGGING_FACE_HUB_TOKEN"):
    login(token=token)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- 1. CRITICAL CONFIG PATCH ---
# We monkey-patch the config here to ensure head_dim is 256. 
# This prevents the "2048 vs 2304" mismatch error even if you haven't edited PaliGemma2.py yet.
def get_fixed_config():
    config = PaliGemma2Config()
    
    # Force head_dim to 256 (matches standard Gemma 2 2B weights)
    # 8 heads * 256 dim = 2048 projection size
    if not hasattr(config.text_config, 'head_dim'):
        setattr(config.text_config, 'head_dim', 256)
    else:
        config.text_config.head_dim = 256
        
    return config

# --- 2. Loader Logic (Using load_state_dict) ---

def load_weights_standard(model, directory):
    """
    Loads .safetensors using standard load_state_dict(strict=False).
    This is safer and reports missing keys accurately.
    """
    files = list(Path(directory).glob("*.safetensors"))
    if not files: 
        raise FileNotFoundError(f"No safetensors found in {directory}")
    
    print(f"Aggregating weights from {len(files)} files...")
    full_state_dict = {}
    
    # 1. Load all tensors into CPU memory
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                full_state_dict[key] = f.get_tensor(key)

    print("Loading state dict into model...")
    
    # 2. Load into model
    # strict=False allows for buffer mismatches (like rotary inv_freq) usually harmless
    missing_keys, unexpected_keys = model.load_state_dict(full_state_dict, strict=False)
    
    # 3. Report Health
    # Filter out "inv_freq" which are Rotary Embedding buffers often recalculated on fly
    significant_missing = [k for k in missing_keys if "inv_freq" not in k]
    
    if significant_missing:
        print(f"\n⚠️ WARNING: {len(significant_missing)} keys were missing!")
        print(f"First 5 missing: {significant_missing[:5]}")
    else:
        print("\n✅ All model weights loaded successfully.")

    if unexpected_keys:
        print(f"ℹ️ Note: {len(unexpected_keys)} unexpected keys in file (usually fine).")

    # Cleanup memory
    del full_state_dict
    gc.collect()
    torch.cuda.empty_cache()

def get_model_and_processor():
    """Setup logic."""
    if not os.path.exists(LOCAL_DIR):
        print(f"Downloading {HF_REPO} to {LOCAL_DIR}...")
        snapshot_download(repo_id=HF_REPO, local_dir=LOCAL_DIR, 
                          allow_patterns=["*.safetensors", "tokenizer.json", "special_tokens_map.json"])

    # 1. Init Model with FIXED config
    print("Initializing Model Architecture...")
    config = get_fixed_config() 
    
    # Initialize on Meta device to save RAM before loading weights, if possible, 
    # but for simplicity/safety we init on CPU then move.
    model = PaliGemma2ForConditionalGeneration(config)
    
    # 2. Load Weights using the new function
    load_weights_standard(model, LOCAL_DIR)
    
    # 3. Move to Device and Cast
    model.to(DTYPE).to(DEVICE)
    model.eval()
    
    # 4. Init Processor
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, padding_side="right")
    
    # Calculate image tokens: (224 / 14)^2 = 256
    num_image_tokens = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    
    processor = PaliGemma2Processor(
        tokenizer, 
        image_tokens=num_image_tokens,
        image_size=config.vision_config.image_size
    )
    
    return model, processor

# --- 3. Corrected Inference Loop ---

@torch.no_grad()
def generate(model, processor, image, prompt, max_tokens=100, temp=0.7, top_p=0.9):
    # 1. Preprocess
    inputs = processor(text=prompt, image=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE).to(DTYPE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    kv_cache = KVCache()
    generated_ids = []
    
    # Initial forward pass inputs
    curr_input_ids = input_ids
    curr_pixel_values = pixel_values
    
    print("Starting generation...")

    # 2. Generation Loop
    for step in range(max_tokens):
        # Forward pass
        # Note: model.forward handles the merging of embeddings internally
        # Logic: If kv_cache is empty, it processes the full prompt.
        # If kv_cache has items, it processes only the last token (curr_input_ids).
        
        outputs = model(
            input_ids=curr_input_ids,
            pixel_values=curr_pixel_values, 
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        
        next_token_logits = outputs["logits"][:, -1, :] # (B, Vocab)

        # Sampling logic
        if temp > 0:
            # Apply Temperature
            next_token_logits = next_token_logits / temp
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        
        # Stop condition
        if token_id == model.config.text_config.eos_token_id:
            break
            
        generated_ids.append(token_id)
        
        # Prepare for next step
        curr_input_ids = next_token # (1, 1)
        
        # Extend attention mask
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=DEVICE, dtype=attention_mask.dtype)], 
            dim=1
        )
        
        # IMPORTANT: Pixel values are only needed for the PREFILL step (when cache is empty).
        # In subsequent steps, we pass None or dummy because the vision embeddings 
        # are already merged into the KV Cache of the first layer.
        # However, keeping it doesn't hurt if the model handle it, but for efficiency/correctness:
        # The prompt processing step embedded the image. The decoding steps just need text.
        
    decoded_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded_text

# --- 4. UI ---

def main():
    # Load model once on startup
    model, processor = get_model_and_processor()
    
    def run_inference(image, text, max_new, temp):
        if image is None: 
            return "Please upload an image."
        if not text: 
            text = "describe this image"
            
        try:
            result = generate(model, processor, image, text, int(max_new), float(temp))
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"

    with gr.Blocks(title="PaliGemma2 Zoo") as app:
        gr.Markdown(f"### PaliGemma2 (3B) on {DEVICE.upper()}")
        gr.Markdown("Custom Implementation with `load_state_dict`")
        
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Input Image")
                prompt = gr.Textbox(label="Prompt", value="caption en", placeholder="e.g. 'caption en' or 'detect cat'")
                
                with gr.Accordion("Advanced Settings", open=False):
                    tokens = gr.Slider(10, 500, 100, step=10, label="Max New Tokens")
                    temp = gr.Slider(0.0, 1.5, 0.0, step=0.1, label="Temperature (0.0 = Greedy)")
                
                btn = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                out = gr.Textbox(label="Output", lines=5)
        
        btn.click(run_inference, [img, prompt, tokens, temp], out)

    print(f"Launching on http://0.0.0.0:7860")
    app.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()