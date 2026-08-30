import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download, login
from dotenv import load_dotenv
import gradio as gr

from transformers import PreTrainedTokenizerBase, CLIPTokenizer
from Zoo.StableDiffusion.SubModels.DDPM import DDPMSampler
from Zoo.StableDiffusion.utils.model_loader import load_models_from_standard_weights

# Environment Setup
load_dotenv()
if token := os.getenv("HUGGING_FACE_HUB_TOKEN"):
    login(token=token)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


device = "cuda" if torch.cuda.is_available() else "cpu"
idle_device = "cpu" if device == "cuda" else None

# =================================================================================
# Helper Functions
# =================================================================================


def rescale(
    x: torch.Tensor, old_range: tuple, new_range: tuple, clamp: bool = False
) -> torch.Tensor:
    """
    Rescales a tensor from an old value range to a new value range.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: int) -> torch.Tensor:
    """
    Generates sinusoidal time embeddings for the given timestep.
    This creates the 320-dimensional raw time embedding expected by the UNet.
    """
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # Concatenate cos and sin -> Shape: (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def load_model(model):
    model.to(device)


def unload_model(model):
    if idle_device:
        model.to(idle_device)


# =================================================================================
# Main Generation Pipeline
# =================================================================================


def generate(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    uncond_prompt: str = "",
    input_image: Optional[Image.Image] = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    n_inference_steps: int = 50,
    models: dict = {},
    seed: Optional[int] = None,
):
    """
    Executes the Stable Diffusion Image Generation Pipeline.
    Supports both Text-to-Image (if input_image is None) and Image-to-Image.
    """
    with torch.no_grad():
        # ---------------------------------------------------------------------------
        # Step 0: Initial Setup & Memory Management Helpers
        # ---------------------------------------------------------------------------
        if not (0 < strength <= 1.0):
            raise ValueError("Strength must be between 0.0 and 1.0")

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for text encoding.")

        # Set the global random seed for reproducibility (if provided)
        if seed is not None:
            torch.manual_seed(seed)

        # Define dimensions (Stable Diffusion v1.5 native resolution is 512x512)
        WIDTH, HEIGHT = 512, 512
        LATENTS_WIDTH, LATENTS_HEIGHT = WIDTH // 8, HEIGHT // 8
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # ---------------------------------------------------------------------------
        # Step 1: Text Encoding (CLIP)
        # ---------------------------------------------------------------------------
        clip = models["clip"]
        load_model(clip)

        # Tokenize and encode the main positive prompt
        cond_tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        cond_context = clip(cond_tokens)  # (1, 77, 768)

        if do_cfg:
            # Tokenize and encode the negative/unconditional prompt
            uncond_tokens = tokenizer(
                uncond_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            uncond_context = clip(uncond_tokens)  # (1, 77, 768)

            # Concatenate to process both simultaneously in the UNet -> (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            context = cond_context

        # Text encoding is done. Free up VRAM.
        unload_model(clip)

        # ---------------------------------------------------------------------------
        # Step 2: Sampler Initialization
        # ---------------------------------------------------------------------------
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
            

        sampler = DDPMSampler(generator=generator)
        sampler.set_inference_timesteps(n_inference_steps)

        # ---------------------------------------------------------------------------
        # Step 3: Latent Initialization (Img2Img vs Txt2Img)
        # ---------------------------------------------------------------------------
        if input_image is not None:
            # --- IMAGE TO IMAGE ---
            vae = models["vae"]
            load_model(vae)

            # Preprocess the PIL Image into a Tensor
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)

            # (H, W, C) -> Float Tensor
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )

            # Scale from [0, 255] to [-1, 1]
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # Reshape to (Batch, Channel, Height, Width) -> (1, 3, 512, 512)
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # Generate initial noise and encode the image into latents
            encoder_noise = torch.randn(latents_shape, device=device)
            latents = vae.encode(input_image_tensor, encoder_noise)

            # Add noise to the latents based on the 'strength' parameter
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Encoding is done. Free up VRAM.
            unload_model(vae)

        else:
            # --- TEXT TO IMAGE ---
            # Start with pure random gaussian noise
            latents = torch.randn(latents_shape, device=device)

        # ---------------------------------------------------------------------------
        # Step 4: Denoising Loop (UNet)
        # ---------------------------------------------------------------------------
        unet = models["unet"]
        load_model(unet)

        timesteps = tqdm(sampler.timesteps, desc="Denoising Image")

        for i, timestep in enumerate(timesteps):
            # Generate the time embedding for the current step -> (1, 320)
            timestep = int(timestep.item())
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                # Duplicate latents to match the concatenated Context (Positive + Negative)
                # (1, 4, 64, 64) -> (2, 4, 64, 64)
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict the noise residual using the UNet
            # Output: (Batch, 4, 64, 64)
            model_output = unet(model_input, context, time_embedding)

            if do_cfg:
                # Split the output into two parts: one for the positive prompt and one for the negative prompt
                model_output_cond, model_output_uncond = model_output.chunk(2, dim=0)

                # Perform Classifier-Free Guidance
                model_output = model_output_uncond + cfg_scale * (
                    model_output_cond - model_output_uncond
                )

            # Step the sampler backward to remove the predicted noise
            # (1, 4, 64, 64) -> (1, 4, 64, 64)
            latents = sampler.step(timestep, latents, model_output)

        # Denoising is done. Free up VRAM.
        unload_model(unet)

        # ---------------------------------------------------------------------------
        # Step 5: Decoding Latents to Image (VAE Decoder)
        # ---------------------------------------------------------------------------
        vae = models["vae"]
        load_model(vae)

        # Decode Latents back into Image Space
        # (1, 4, 64, 64) -> (1, 3, 512, 512)
        images = vae.decode(latents)

        # Decoding is done. Free up VRAM.
        unload_model(vae)

        # ---------------------------------------------------------------------------
        # Step 6: Image Post-Processing
        # ---------------------------------------------------------------------------
        # Rescale from [-1, 1] back to [0, 255]
        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # Re-order dimensions for Image creation: (Batch, C, H, W) -> (Batch, H, W, C)
        images = images.permute(0, 2, 3, 1)

        # Move to CPU, convert to uint8, and extract the first (and only) image from the batch
        image_array = images.to("cpu", torch.uint8).numpy()[0]

        return image_array

# =================================================================================
# Gradio Interface & Model Initialization
# =================================================================================

# Global state to keep models in memory across UI requests
GLOBAL_MODELS = {}
GLOBAL_TOKENIZER = None

HF_REPO = "runwayml/stable-diffusion-v1-5"
LOCAL_DIR = "./saved_model/StableDiffusion"

def initialize_models():
    """Downloads (if necessary) and loads the tokenizer and model weights."""
    global GLOBAL_MODELS, GLOBAL_TOKENIZER
    
    print(f"Ensuring model weights exist in {LOCAL_DIR}...")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # 1. Download/Verify the specific Checkpoint file
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="v1-5-pruned-emaonly.ckpt",
        local_dir=LOCAL_DIR
    )
    
    # 2. Download/Verify the tokenizer directory
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=["tokenizer/*"],
        local_dir=LOCAL_DIR
    )
    tokenizer_path = os.path.join(LOCAL_DIR, "tokenizer")

    # 3. Load into memory
    print("Loading Tokenizer...")
    GLOBAL_TOKENIZER = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    
    print("Loading Checkpoint Weights... (This may take a minute)")
    GLOBAL_MODELS = load_models_from_standard_weights(ckpt_path, device)
    print("Initialization Complete! Launching UI...")

def gradio_predict(prompt, uncond_prompt, input_image, strength, cfg_scale, steps, seed):
    """Wrapper function to connect Gradio inputs to the generation pipeline."""
    
    # Satisfies strict type-checking for Pylance
    if GLOBAL_TOKENIZER is None:
        raise RuntimeError("Tokenizer was not initialized properly.")
        
    # Handle random seed logic
    active_seed = None if seed == -1 else int(seed)
    
    image_array = generate(
        prompt=prompt,
        tokenizer=GLOBAL_TOKENIZER,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        n_inference_steps=int(steps),
        models=GLOBAL_MODELS,
        seed=active_seed
    )
    
    # Convert output numpy array to PIL Image for Gradio
    return Image.fromarray(image_array)

def launch_app():
    initialize_models()
    
    with gr.Blocks(title="PyTorch Stable Diffusion from Scratch") as demo:
        gr.Markdown(f"# 🎨 Stable Diffusion v1.5 (From Scratch)\nRunning on **{device.upper()}**.")
        gr.Markdown("A clean, native PyTorch implementation with automatic weight downloading to `./saved_model/StableDiffusion`.")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", placeholder="A highly detailed cinematic shot of a cybernetic cat...")
                uncond_prompt = gr.Textbox(label="Negative Prompt", value="lowres, bad anatomy, bad quality, blurry, worst quality")
                
                input_image = gr.Image(label="Input Image (Optional for Img2Img)", type="pil")
                
                with gr.Accordion("Advanced Settings", open=False):
                    strength = gr.Slider(0.01, 1.0, value=0.8, step=0.01, label="Denoising Strength (Img2Img Only)")
                    cfg_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Scale")
                    steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps (DDPM)")
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                generate_btn = gr.Button("Generate Image", variant="primary")
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Output")

        generate_btn.click(
            fn=gradio_predict,
            inputs=[prompt, uncond_prompt, input_image, strength, cfg_scale, steps, seed],
            outputs=[output_image]
        )
        
    demo.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    launch_app()