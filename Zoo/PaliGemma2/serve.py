import os
from pathlib import Path
import sys
import json
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download, login
from utils.LoadModel import load_hf_model
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
HuggingFaceHubToken = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HuggingFaceHubToken:
    print("Error: Hugging Face Hub token not found in environment variables.")
    print("Please set HUGGING_FACE_HUB_TOKEN in your .env file.")
    sys.exit(1)

login(token=HuggingFaceHubToken)
os.environ["HF_TOKEN"] = HuggingFaceHubToken

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
HF_MODEL_ID = "google/paligemma2-3b-pt-224"  # Placeholder: Adjust if using specific Gemma2 weights
LOCAL_SAVE_DIR = "./saved_model/Paligemma2"  # Local directory to save the model snapshot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float32
)


def load_config():
    config_path = os.path.join(LOCAL_SAVE_DIR, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = PaliGemma2Config(config_dict)

    return config


def download_and_load_weights(hf_id: str, local_dir: str, device: str = DEVICE):
    """Ensure weights are available locally (via snapshot_download) and load them
    using `load_hf_model` from `utils.LoadModel`.

    Returns a ready-to-use model (moved to `device`) and tokenizer.
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # If local_dir already contains a config.json, assume it's a local model folder
    local_path = Path(local_dir)
    config_file = local_path / "config.json"

    if not config_file.exists():
        print(
            f"Downloading model snapshot from Hugging Face: {hf_id} -> {local_dir}..."
        )
        try:
            snapshot_download(
                repo_id=hf_id,
                local_dir=local_dir,
                allow_patterns=[
                    "*.safetensors",
                    "config.json",
                    "pytorch_model.bin",
                    "*.bin",
                    "*.json",
                ],
            )  # downloads into local_dir
        except Exception as e:
            print(f"Error downloading model snapshot: {e}")
            raise

    # Use the loader utility to create model and tokenizer
    model, tokenizer = load_hf_model(local_dir, device=device)
    return model, tokenizer


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
            kv_cache=kv_cache,
        )

        logits = outputs["logits"]  # (B, Seq, Vocab)
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
        curr_input_ids = next_token  # Pass only the new token

        # Update attention mask for the new token (simple logic: append 1)
        # Note: Your model's _merge logic handles position IDs based on cache presence
        # so explicit mask expansion logic might be handled internally or via the mask passed.
        # For simple generation, we often just pass the new token ID.
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), device=DEVICE, dtype=attention_mask.dtype),
            ],
            dim=1,
        )

    # 4. Decode Output
    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text


# --- Main Application ---


def main():
    print("Initializing PaliGemma2 Service...")

    # 1. Load Configuration
    config = load_config()

    # 2. Download and load model + tokenizer
    print("Downloading and loading model (weights + config)...")
    try:
        model, returned_tokenizer = download_and_load_weights(
            HF_MODEL_ID, LOCAL_SAVE_DIR, device=DEVICE
        )
    except Exception as e:
        print(f"Error loading weights: {e}")
        print(
            "Make sure you have access to the HF model and the HF_MODEL_ID is correct."
        )
        raise

    model.to(DEVICE).to(DTYPE)
    model.eval()

    # 3. Initialize Processor using tokenizer returned from the model loader
    print("Initializing processor with tokenizer from model...")
    processor = PaliGemma2Processor(
        tokenizer=returned_tokenizer,
        image_tokens=config.vision_config.patch_size,
        image_size=config.vision_config.image_size,
    )

    # 5. Gradio Interface
    def predict(image, text, max_tokens, temp):
        if image is None:
            return "Please upload an image."
        if not text:
            text = "describe this image"  # default prompt

        try:
            result = generate_text(
                model=model,
                processor=processor,
                image=image,
                prompt=text,
                max_new_tokens=int(max_tokens),
                temperature=float(temp),
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
                input_text = gr.Textbox(
                    label="Prompt", placeholder="Describe this image...", lines=2
                )

                with gr.Accordion("Advanced Settings", open=False):
                    slider_tokens = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max New Tokens",
                    )
                    slider_temp = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )

                submit_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(label="Model Output", interactive=False)

        submit_btn.click(
            fn=predict,
            inputs=[input_image, input_text, slider_tokens, slider_temp],
            outputs=[output_text],
        )

    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
