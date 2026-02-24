import os
import sys
import torch
import gradio as gr
from pathlib import Path
from huggingface_hub import snapshot_download, login
from safetensors import safe_open
import gc
from dotenv import load_dotenv

# Navigates up until it finds the directory containing 'Zoo'
current_path = Path(__file__).resolve()
root_node = next(p for p in current_path.parents if (p / "Zoo").exists())

if str(root_node) not in sys.path:
    sys.path.insert(0, str(root_node))

# Environment Setup
load_dotenv()
# Ensure we can import from the current directory (Zoo/Mistral3)
sys.path.append(os.path.dirname(__file__))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)

# Import the specific configurations and the wrapper from your new file
from Ministral3Multimodal import (
    Ministral3MultimodalConfig,
    Ministral3ForConditionalGeneration
)
from Zoo.Ministral3.SubModels.Ministral3 import Ministral3Config, RopeParameters
from Zoo.Ministral3.SubModels.Pixtral import PixtralConfig

from transformers import AutoProcessor

# --- Constants ---
HF_REPO = "mistralai/Ministral-3-3B-Instruct-2512"
LOCAL_DIR = "./saved_model/Ministral3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Mistral-3 is optimized for bfloat16
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

# No Need For Authentication
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Robust Loader with Remapping ---

def remap_key(key):
    """
    Maps Hugging Face checkpoint keys to our Custom Model keys.
    
    HF Checkpoint Structure:
      - vision_tower.*
      - multi_modal_projector.*
      - language_model.model.*  (Backbone)
      - language_model.lm_head.* (Head)
    
    Custom Model Structure:
      - model.vision_tower.*
      - model.multi_modal_projector.*
      - model.language_model.*  (Backbone)
      - lm_head.*
    """
    # 1. Map Text Backbone: language_model.model -> model.language_model
    if key.startswith("language_model.model."):
        return key.replace("language_model.model.", "model.language_model.")
    
    # 2. Map Vision Tower: vision_tower -> model.vision_tower
    if key.startswith("vision_tower."):
        return f"model.{key}"

    # 3. Map Projector: multi_modal_projector -> model.multi_modal_projector
    if key.startswith("multi_modal_projector."):
        return f"model.{key}"

    # 4. Map LM Head (if present inside language_model)
    if key.startswith("language_model.lm_head."):
        return key.replace("language_model.lm_head.", "lm_head.")
    
    # 5. Fallback for unexpected keys, try prepending model.
    if not key.startswith("model.") and not key.startswith("lm_head."):
         return f"model.{key}"

    return key


def load_weights_into_model(model, directory, device):
    """Streams .safetensors directly into the model object with remapping."""
    files = list(Path(directory).glob("*.safetensors"))
    if not files: raise FileNotFoundError(f"No safetensors found in {directory}")
    
    print(f"Loading {len(files)} weight files...")
    
    # Get all valid keys from our model to check against
    model_keys = set(model.state_dict().keys())
    loaded_keys = set()
    
    for file in files:
        with safe_open(file, framework="pt", device=DEVICE) as f:
            for file_key in f.keys():
                # Apply remapping
                custom_key = remap_key(file_key)
                
                if custom_key not in model_keys:
                    # Optional: Print only if it's NOT just a mismatch we caused
                    print(f"Skipping {file_key} -> {custom_key} (not in model)")
                    continue
                
                # Load -> Cast -> Assign
                tensor = f.get_tensor(file_key).to(device=device, dtype=DTYPE)
                _set_nested_param(model, custom_key, tensor)
                loaded_keys.add(custom_key)
                del tensor
        gc.collect()
        torch.cuda.empty_cache()

    # Verify loading
    missing_keys = model_keys - loaded_keys
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys were not loaded!")
        # Optional: inspect specific missing keys
        # print(missing_keys)
    else:
        print("All model weights loaded successfully.")

def _set_nested_param(model, key, tensor):
    """Helper to traverse nested submodules and assign data."""
    try:
        module_name, param_name = key.rsplit(".", 1) if "." in key else ("", key)
        submodule = model.get_submodule(module_name) if module_name else model
        param = getattr(submodule, param_name)
        
        # Handle shape mismatches seamlessly
        if param.shape != tensor.shape and param.numel() == tensor.numel():
            tensor = tensor.view(param.shape)
            
        with torch.no_grad():
            param.data = tensor
    except Exception as e:
        print(f"Warning: Failed to load {key}: {e}")

def get_model_and_processor():
    """Setup logic."""
    print(f"Downloading {HF_REPO} to {LOCAL_DIR}...")
    snapshot_download(
        repo_id=HF_REPO, 
        local_dir=LOCAL_DIR, 
        allow_patterns=["model.safetensors"]
    )

    print("Initializing Model Architecture...")
    
    # 1. Map Text Config for the 3B model
    rope_params = RopeParameters(
        beta_fast=32.0,
        beta_slow=1.0,
        factor=16.0,
        llama_4_scaling_beta=0.1,
        mscale=1.0,
        mscale_all_dim=1.0,
        original_max_position_embeddings=16384,
        rope_theta=1000000.0,
        rope_type="yarn",
        type="yarn"
    )

    text_config = Ministral3Config(
        attention_dropout=0.0,
        head_dim=128,
        hidden_size=3072,           # Updated from 4096
        intermediate_size=9216,     # Updated from 14336
        max_position_embeddings=262144,
        num_attention_heads=32,
        num_hidden_layers=26,       # Updated from 34
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        vocab_size=131072,
        tie_word_embeddings=True,   # Updated from False
        rope_parameters=rope_params.__dict__
    )

    # 2. Map Vision Config (Dimensions remain similar, but added rope_params block)
    vision_config = PixtralConfig(
        attention_dropout=0.0,
        head_dim=64,
        hidden_size=1024,
        image_size=1540,
        intermediate_size=4096,
        num_attention_heads=16,
        num_channels=3,
        num_hidden_layers=24,
        patch_size=14,
    )

    # 3. Combine into the Multimodal wrapper
    multimodal_config = Ministral3MultimodalConfig(
        spatial_merge_size=2,
        image_token_index=10,
        vision_feature_layer=-1,
        tie_word_embeddings=True,   # Updated to match text_config
        text_config=text_config,
        vision_config=vision_config
    )

    model = Ministral3ForConditionalGeneration(multimodal_config).to(DEVICE).to(DTYPE)
    
    # Load Weights
    load_weights_into_model(model, LOCAL_DIR, DEVICE)
    model.eval()
    
    # Init Processor (Mistral processors handle image patching and text formatting)
    processor = AutoProcessor.from_pretrained(model) # Since Preprocessor not Found in the repo on hugging face, we rely on AutoProcessor to find the right one based on the model's config.
    
    return model, processor

# --- Inference ---

@torch.no_grad()
def generate(model, processor, image, prompt, max_tokens=100, temp=0.7):
    # 1. Preprocess using Mistral's Chat Template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Format the prompt automatically 
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process text and image
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)
    
    # Processor might output image sizes; if not, the model defaults appropriately
    image_sizes = inputs.get("image_sizes", None)
    if image_sizes is not None:
        image_sizes = image_sizes.to(DEVICE)

    generated_ids = []

    # --- Prefill Step ---
    # Pass input_ids and pixel_values. The model automatically initializes KVCache if None.
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        past_key_values=None 
    )
    
    # Get logits for the VERY LAST token of the prompt
    next_logits = outputs["logits"][:, -1, :]
    kv_cache = outputs["past_key_values"]

    # --- Decode Loop ---
    for _ in range(max_tokens):
        # Sample the next token
        if temp > 0:
            probs = torch.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        
        # Check for EOS token
        if token_id == processor.tokenizer.eos_token_id:
            break
            
        generated_ids.append(token_id)
        
        # Call model with ONLY the new token. pixel_values are None to skip vision encoder.
        outputs = model(
            input_ids=next_token,
            pixel_values=None,   # IMPORTANT: Skip vision in decode
            image_sizes=None,
            past_key_values=kv_cache
        )
        
        next_logits = outputs["logits"][:, -1, :]
        kv_cache = outputs["past_key_values"]

    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

# --- UI ---

def main():
    print("Starting Ministral-3 3b Multimodal Server...")
    model, processor = get_model_and_processor()
    
    def run_inference(image, text, max_new, temp):
        if not image: return "Upload an image."
        text = text or "Describe this image in detail."
        try:
            return generate(model, processor, image, text, int(max_new), float(temp))
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Blocks(title="Ministral-3 Zoo") as app:
        gr.Markdown(f"### Ministral-3 (3b) Multimodal on {DEVICE.upper()}")
        with gr.Row():
            img = gr.Image(type="pil", label="Image")
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="Describe this image in detail.")
                tokens = gr.Slider(10, 1024, 256, label="Max Tokens")
                temp = gr.Slider(0.0, 1.5, 0.7, label="Temperature")
                btn = gr.Button("Generate", variant="primary")
                out = gr.Textbox(label="Output", lines=5)
        
        btn.click(run_inference, [img, prompt, tokens, temp], out)

    app.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()