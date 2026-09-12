import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .SubModels.AutoEncoderKL import AutoencoderKL
from .SubModels.FluxTransformer2DModel import FluxTransformer2DModel
from .SubModels.SchedulingFlowMatchEulerDiscrete import FlowMatchEulerDiscreteScheduler
from .SubModels.T5EncoderModel import T5EncoderModel
from .pipeline import FluxPipeline
from .utils.model_loader import (
    load_flux_transformer_weights,
    load_flux_vae_weights,
    load_flux_t5_weights,
)


def load_pipeline(checkpoint_path: str = "black-forest-labs/FLUX.1-schnell", device: str = "cuda"):
    print("==================================================")
    print("Initializing FLUX.1 [schnell] Architecture...")
    print("==================================================")

    # 1. Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        time_shift_type="linear",
    )

    # 2. VAE
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        scaling_factor=0.3611,
        shift_factor=0.1159,
    )
    load_flux_vae_weights(vae, repo_id=checkpoint_path, device="cpu")

    # 3. CLIP Text Encoder & Tokenizer
    print("Loading CLIP Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")

    # 4. T5-XXL Text Encoder & Tokenizer
    print("Loading T5-XXL Tokenizer...")
    tokenizer_2 = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel()
    load_flux_t5_weights(text_encoder_2, repo_id=checkpoint_path, device="cpu")

    # 5. Flux Transformer 2D Model (12B params)
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        num_layers=19,
        num_single_layers=38,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=False,
    )
    load_flux_transformer_weights(transformer, repo_id=checkpoint_path, device="cpu")

    # Pipeline
    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
    )

    print(f"Moving models to {device} in bfloat16...")
    pipe.to(device, dtype=torch.bfloat16)
    print("FLUX.1 [schnell] loaded completely and ready!")
    return pipe


if __name__ == "__main__":
    prompt = "A sleek cybernetic robotic tiger walking through a rain-slicked Tokyo street at night, neon reflections, 8k resolution, cinematic lighting"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device=device)

    print(f"\nGenerating image for: '{prompt}'...")
    generator = torch.Generator(device=device).manual_seed(42)
    images = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        generator=generator,
    )

    output_filename = "flux_output.png"
    images[0].save(output_filename)
    print(f"Saved generated image to {output_filename}!")