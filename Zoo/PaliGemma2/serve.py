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
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

# Authentication
if token := os.getenv("HUGGING_FACE_HUB_TOKEN"):
    login(token=token)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Lightweight Loader ---

def load_weights_into_model(model, directory, device):
    """Streams .safetensors directly into the model object."""
    files = list(Path(directory).glob("*.safetensors"))
    if not files: raise FileNotFoundError(f"No safetensors found in {directory}")
    
    print(f"Loading {len(files)} weight files...")
    state_dict_keys = set(model.state_dict().keys())
    
    for file in files:
        with safe_open(file, framework="pt", device=DEVICE) as f:
            for key in f.keys():
                if key in state_dict_keys:
                    # Load -> Move to GPU -> Cast -> Assign
                    tensor = f.get_tensor(key).to(device=device, dtype=DTYPE)
                    _set_nested_param(model, key, tensor)
                    del tensor
        gc.collect()
        torch.cuda.empty_cache()

def _set_nested_param(model, key, tensor):
    """Helper to traverse model.layers.0.weight and assign data."""
    try:
        module_name, param_name = key.rsplit(".", 1) if "." in key else ("", key)
        submodule = model.get_submodule(module_name) if module_name else model
        param = getattr(submodule, param_name)
        
        # Handle shape mismatches (e.g. squeezed tensors)
        if param.shape != tensor.shape and param.numel() == tensor.numel():
            tensor = tensor.view(param.shape)
            
        with torch.no_grad():
            param.data = tensor
    except Exception as e:
        print(f"Warning: Failed to load {key}: {e}")

def get_model_and_processor():
    """Setup logic."""
    print(f"Downloading {HF_REPO} to {LOCAL_DIR}...")
    snapshot_download(repo_id=HF_REPO, local_dir=LOCAL_DIR, 
                      allow_patterns=["*.safetensors", "tokenizer.json", "special_tokens_map.json"])

    # 1. Init Model (Architecture defined in code, not JSON)
    print("Initializing Model Architecture...")
    config = PaliGemma2Config() 
    model = PaliGemma2ForConditionalGeneration(config).to(DEVICE).to(DTYPE)
    
    # 2. Load Weights
    load_weights_into_model(model, LOCAL_DIR, DEVICE)
    model.eval()
    
    # 3. Init Processor
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, padding_side="right")
    processor = PaliGemma2Processor(tokenizer, 
                                    image_tokens=config.vision_config.patch_size, # Actually num_patches (256) implicitly handled
                                    image_size=config.vision_config.image_size) # 224
    
    # Fix: PaliGemma uses 256 tokens for 224px image (14x14 patch) -> (224/14)^2 = 256
    processor.image_tokens = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
    
    return model, processor

# --- Inference ---

@torch.no_grad()
def generate(model, processor, image, prompt, max_tokens=100, temp=0.7):
    # 1. Preprocess
    inputs = processor(text=prompt, image=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE).to(DTYPE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    kv_cache = KVCache()
    generated_ids = []

    # --- Prefill Step ---
    # Process the image and full prompt to populate the KV cache
    prefill_outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache
    )
    
    # Get logits for the VERY LAST token of the prompt
    next_logits = prefill_outputs["logits"][:, -1, :]

    # --- Decode Loop ---
    for _ in range(max_tokens):
        # Sample the next token
        if temp > 0:
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        if next_token.item() == model.config.text_config.eos_token_id:
            break
            
        generated_ids.append(next_token.item())
        
        # Prepare inputs for the next decoding step
        # We only need the new token and an updated attention mask
        curr_input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=DEVICE)], dim=1)

        # Call model with ONLY the new token. pixel_values are None.
        decode_outputs = model(
            input_ids=curr_input_ids,
            pixel_values=None, # IMPORTANT
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        next_logits = decode_outputs["logits"][:, -1, :]

    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

# --- UI ---

def main():
    model, processor = get_model_and_processor()
    model.tie_weights()  # Ensure tied weights are correctly set after loading
    
    def run_inference(image, text, max_new, temp):
        if not image: return "Upload an image."
        text = text or "describe this image"
        try:
            return generate(model, processor, image, text, int(max_new), float(temp))
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title="PaliGemma2 Zoo") as app:
        gr.Markdown(f"### PaliGemma2 (3B) on {DEVICE.upper()}")
        with gr.Row():
            img = gr.Image(type="pil", label="Image")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="describe this image")
                tokens = gr.Slider(10, 500, 100, label="Max Tokens")
                temp = gr.Slider(0.0, 1.5, 0.7, label="Temperature")
                btn = gr.Button("Generate", variant="primary")
                out = gr.Textbox(label="Output")
        
        btn.click(run_inference, [img, prompt, tokens, temp], out)

    app.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()