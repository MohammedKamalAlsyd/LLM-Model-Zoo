import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .SubModels.AutoEncoderKL import AutoencoderKL
from .SubModels.FluxTransformer2DModel import FluxTransformer2DModel
from .SubModels.SchedulingFlowMatchEulerDiscrete import FlowMatchEulerDiscreteScheduler
from .SubModels.T5EncoderModel import T5EncoderModel
from .pipeline import FluxPipeline


def load_pipeline(checkpoint_path: str = "black-forest-labs/FLUX.1-schnell", device: str = "cuda"):
    print("Loading FLUX.1 [schnell] components...")

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

    # 2. AutoEncoder (VAE)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        scaling_factor=0.3611,
        shift_factor=0.1159,
    )

    # 3. CLIP Text Encoder & Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")

    # 4. T5-XXL Text Encoder & Tokenizer (using AutoTokenizer to resolve fast tokenizer cleanly)
    tokenizer_2 = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel()

    # 5. Flux Transformer 2D Model
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

    # Initialize the Pipeline
    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
    )

    pipe.to(device, dtype=torch.bfloat16)
    print("FLUX.1 [schnell] loaded successfully!")
    return pipe


if __name__ == "__main__":
    prompt = "A sleek cybernetic robotic tiger walking through a rain-slicked Tokyo street at night, neon reflections, 8k resolution, cinematic lighting"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device=device)

    print(f"Generating image for prompt: '{prompt}'...")
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