import collections
import gc
from typing import Dict, List, Optional, Tuple, cast
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .SubModels.AutoEncoderKL import AutoencoderKL, DecoderOutput
from .SubModels.FluxTransformer2DModel import FluxTransformer2DModel, Transformer2DModelOutput
from .SubModels.SchedulingFlowMatchEulerDiscrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from .SubModels.T5EncoderModel import T5EncoderModel
from .pipeline import calculate_shift, pack_latents, prepare_latent_image_ids, unpack_latents
from .utils.model_loader import _set_module_tensor, get_safetensors_files, stream_safetensors_to_model


def purge_memory():
    """Forces aggressive garbage collection and releases cached CUDA blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _free_submodule(module: nn.Module):
    """Replaces all CUDA parameters in a module with zero-byte meta parameters to free VRAM."""
    for name, param in module.named_parameters():
        meta_param = nn.Parameter(
            torch.empty(param.shape, device="meta", dtype=param.dtype),
            requires_grad=False,
        )
        _set_module_tensor(module, name, meta_param)


class StreamingFluxPipeline:
    """
    Kaggle T4-optimized FLUX.1 [schnell] pipeline.
    Uses sequential block-by-block streaming to stay under 1.5 GB VRAM and 2.5 GB System RAM.
    """

    def __init__(self, checkpoint_path: str = "black-forest-labs/FLUX.1-schnell", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.vae_scale_factor = 8

        print("Locating repository shards on Hugging Face Hub / local cache...")
        self.transformer_files = get_safetensors_files(checkpoint_path, subfolder="transformer")
        self.t5_files = get_safetensors_files(checkpoint_path, subfolder="text_encoder_2")
        self.vae_files = get_safetensors_files(checkpoint_path, subfolder="vae")

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            time_shift_type="linear",
        )

    # --------------------------------------------------------------------------
    # STAGE 1: Text Encoding (T5 fits in 9.5 GB on T4, then purges)
    # --------------------------------------------------------------------------
    def _encode_prompt(
        self, prompt: str, max_sequence_length: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("\n[Stage 1/3] Loading CLIP-L and T5-XXL for Text Encoding...")

        # 1. CLIP-L Pooled Encoding
        tokenizer = CLIPTokenizer.from_pretrained(self.checkpoint_path, subfolder="tokenizer")
        raw_clip = CLIPTextModel.from_pretrained(
            self.checkpoint_path, subfolder="text_encoder", torch_dtype=self.dtype
        )
        text_encoder = cast(nn.Module, raw_clip).to(self.device)

        clip_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_output = text_encoder(input_ids=clip_inputs["input_ids"].to(self.device))
        pooled_prompt_embeds = getattr(clip_output, "pooler_output", clip_output[1]).to(
            dtype=self.dtype, device=self.device
        )

        del text_encoder, tokenizer, raw_clip
        purge_memory()

        # 2. T5-XXL Token Encoding (~9.5 GB)
        tokenizer_2 = AutoTokenizer.from_pretrained(self.checkpoint_path, subfolder="tokenizer_2")
        with torch.device("meta"):
            t5_encoder = T5EncoderModel()

        print("  Streaming T5-XXL directly to GPU...")
        stream_safetensors_to_model(t5_encoder, self.t5_files, device=self.device, dtype=self.dtype)

        t5_inputs = tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_output = t5_encoder(input_ids=t5_inputs["input_ids"].to(self.device))
        prompt_embeds = t5_output.last_hidden_state.to(dtype=self.dtype, device=self.device)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        del t5_encoder, tokenizer_2, t5_output
        purge_memory()
        print("  Prompt encoded. Text encoders purged. VRAM freed.")
        return prompt_embeds, pooled_prompt_embeds, txt_ids

    # --------------------------------------------------------------------------
    # STAGE 2: Block-by-Block Transformer Denoising (Peaks at only ~1.5 GB VRAM!)
    # --------------------------------------------------------------------------
    def _denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        num_inference_steps: int = 4,
    ) -> torch.Tensor:
        print("\n[Stage 2/3] Initializing Sequential Block Streaming for Transformer...")

        # 1. Open safetensors memory-mapped handles
        open_handles = {p: safe_open(p, framework="pt", device="cpu") for p in self.transformer_files}

        # 2. Index all weight keys into static, dual-stream, and single-stream groups
        static_weights: List[Tuple[str, str]] = []
        dual_blocks_map: Dict[int, List[Tuple[str, str, str]]] = collections.defaultdict(list)
        single_blocks_map: Dict[int, List[Tuple[str, str, str]]] = collections.defaultdict(list)

        for fpath, handle in open_handles.items():
            for key in handle.keys():
                if key.startswith("transformer_blocks."):
                    parts = key.split(".", 2)
                    dual_blocks_map[int(parts[1])].append((fpath, key, parts[2]))
                elif key.startswith("single_transformer_blocks."):
                    parts = key.split(".", 2)
                    single_blocks_map[int(parts[1])].append((fpath, key, parts[2]))
                else:
                    static_weights.append((fpath, key))

        # 3. Build model on meta device
        with torch.device("meta"):
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

        # 4. Load static layers (< 100 MB) onto GPU once
        for fpath, key in static_weights:
            tensor = open_handles[fpath].get_tensor(key).to(device=self.device, dtype=self.dtype)
            _set_module_tensor(transformer, key, tensor)
            del tensor

        print("  Static layers loaded onto GPU (<100 MB). Starting Euler flow matching...")

        # Schedule Setup
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len)
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=self.device, mu=mu)
        self.scheduler.set_begin_index(0)

        # 5. Denoising Loop
        for step_idx, t in enumerate(self.scheduler.timesteps):
            print(f"  Denoising step {step_idx + 1}/{num_inference_steps} (timestep {t.item():.1f})...")
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with torch.no_grad():
                # A. Static input projections
                h_states = transformer.x_embedder(latents)
                enc_h_states = transformer.context_embedder(prompt_embeds)
                t_expanded = timestep * 1000.0
                temb = transformer.time_text_embed(t_expanded, pooled_prompt_embeds)

                ids = torch.cat((txt_ids, img_ids), dim=0)
                image_rotary_emb = transformer.pos_embed(ids)

                # B. Stream 19 Dual-Stream Blocks (one block in VRAM at a time)
                for i in range(len(transformer.transformer_blocks)):
                    block = transformer.transformer_blocks[i]
                    for fpath, full_key, subkey in dual_blocks_map[i]:
                        tensor = open_handles[fpath].get_tensor(full_key).to(device=self.device, dtype=self.dtype)
                        _set_module_tensor(block, subkey, tensor)
                        del tensor

                    enc_h_states, h_states = block(
                        hidden_states=h_states,
                        encoder_hidden_states=enc_h_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                    _free_submodule(block)

                # C. Stream 38 Single-Stream Blocks (one block in VRAM at a time)
                for j in range(len(transformer.single_transformer_blocks)):
                    s_block = transformer.single_transformer_blocks[j]
                    for fpath, full_key, subkey in single_blocks_map[j]:
                        tensor = open_handles[fpath].get_tensor(full_key).to(device=self.device, dtype=self.dtype)
                        _set_module_tensor(s_block, subkey, tensor)
                        del tensor

                    enc_h_states, h_states = s_block(
                        hidden_states=h_states,
                        encoder_hidden_states=enc_h_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                    _free_submodule(s_block)

                # D. Static output projections
                h_states = transformer.norm_out(h_states, temb)
                model_output = transformer.proj_out(h_states)

                # E. Euler Integration Step
                step_output = self.scheduler.step(model_output, t, latents, return_dict=True)
                if isinstance(step_output, FlowMatchEulerDiscreteSchedulerOutput):
                    latents = step_output.prev_sample
                else:
                    latents = step_output[0]

        del open_handles, transformer
        purge_memory()
        print("  Denoising complete. Transformer purged from memory.")
        return latents

    # --------------------------------------------------------------------------
    # STAGE 3: VAE Decoding (~335 MB, easily fits on T4)
    # --------------------------------------------------------------------------
    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> Image.Image:
        print("\n[Stage 3/3] Streaming VAE to GPU and decoding final image...")

        with torch.device("meta"):
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=16,
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                scaling_factor=0.3611,
                shift_factor=0.1159,
            )

        stream_safetensors_to_model(vae, self.vae_files, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            latents = unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = vae.unscale_latents(latents)
            dec_output = vae.decode(latents, return_dict=True)
            sample = dec_output.sample if isinstance(dec_output, DecoderOutput) else dec_output

            images = (sample / 2.0 + 0.5).clamp(0.0, 1.0)
            images_np = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images_uint8 = (images_np * 255.0).round().astype("uint8")

        del vae
        purge_memory()
        print("  Decoding finished. Image generated successfully.")
        return Image.fromarray(images_uint8[0])

    # --------------------------------------------------------------------------
    # Pipeline Call
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        height = 2 * (int(height) // 16) * 8
        width = 2 * (int(width) // 16) * 8

        prompt_embeds, pooled_prompt_embeds, txt_ids = self._encode_prompt(prompt)

        latent_channels = 16
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        noise = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = pack_latents(noise)
        img_ids = prepare_latent_image_ids(height, width, device=self.device, dtype=self.dtype)

        latents = self._denoise_latents(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            num_inference_steps=num_inference_steps,
        )

        image = self._decode_latents(latents, height, width)
        return image


if __name__ == "__main__":
    prompt = (
        "A sleek cybernetic robotic tiger walking through a rain-slicked Tokyo street at night, "
        "neon reflections, 8k resolution, cinematic lighting"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemError("CUDA GPU is required to run FLUX.")

    pipeline = StreamingFluxPipeline(device=device)

    generator = torch.Generator(device=device).manual_seed(42)
    output_image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        generator=generator,
    )

    output_filename = "flux_output.png"
    output_image.save(output_filename)
    print(f"\nCompleted! Saved generated image to {output_filename}")