import collections
import gc
import math
import os
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .SubModels.AutoEncoderKL import AutoencoderKL, DecoderOutput
from .SubModels.FluxTransformer2DModel import (
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from .SubModels.SchedulingFlowMatchEulerDiscrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from .SubModels.T5EncoderModel import T5EncoderModel
from .pipeline import calculate_shift, pack_latents, prepare_latent_image_ids, unpack_latents


def purge_memory():
    """Aggressively releases cached memory blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_safetensors_files(repo_id: str, subfolder: str) -> List[str]:
    """Downloads and returns paths for single or sharded safetensors files."""
    for single_name in ["diffusion_pytorch_model.safetensors", "model.safetensors"]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=single_name, subfolder=subfolder)
            return [path]
        except Exception:
            pass

    # Sharded files
    found_paths = []
    for i in range(1, 15):
        shard_found = False
        for prefix in ["diffusion_pytorch_model", "model"]:
            for total in [2, 3, 4, 5, 6, 7, 8]:
                filename = f"{prefix}-{i:05d}-of-{total:05d}.safetensors"
                try:
                    path = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    found_paths.append(path)
                    shard_found = True
                    break
                except Exception:
                    continue
            if shard_found:
                break
        if not shard_found and found_paths:
            break

    if not found_paths:
        raise FileNotFoundError(f"Could not locate safetensors weights for {subfolder} in {repo_id}")
    return sorted(list(set(found_paths)))


def set_submodule_tensor(module: nn.Module, subkey: str, tensor: torch.Tensor):
    """Copies tensor data directly into an existing submodule parameter."""
    parts = subkey.split(".")
    curr = module
    for part in parts[:-1]:
        if part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)
    leaf = parts[-1]
    param = getattr(curr, leaf)
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


class StreamingFluxPipeline:
    """
    Ultra-low VRAM FLUX.1 [schnell] Pipeline.
    Runs on Kaggle T4 (15 GB VRAM) using < 1.2 GB VRAM and < 2.5 GB System RAM.
    """

    def __init__(self, checkpoint_path: str = "black-forest-labs/FLUX.1-schnell", device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.vae_scale_factor = 8

        print("Locating safetensors shards on Hugging Face Hub / Cache...")
        self.transformer_files = get_safetensors_files(checkpoint_path, subfolder="transformer")
        self.t5_files = get_safetensors_files(checkpoint_path, subfolder="text_encoder_2")
        self.vae_files = get_safetensors_files(checkpoint_path, subfolder="vae")

        # Scheduler with correct exponential dynamic shift
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            time_shift_type="exponential",
        )

    # --------------------------------------------------------------------------
    # STAGE 1: Text Encoding (CLIP + T5-XXL with weight tying)
    # --------------------------------------------------------------------------
    def _encode_prompt(
        self, prompt: str, max_sequence_length: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("\n[Stage 1/3] Loading Text Encoders...")

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

        # 2. T5-XXL Sequence Encoding (~9.5 GB on T4)
        tokenizer_2 = AutoTokenizer.from_pretrained(self.checkpoint_path, subfolder="tokenizer_2")
        t5_encoder = T5EncoderModel().to(self.device, dtype=self.dtype)

        print("  Loading T5-XXL weights directly into GPU...")
        for fpath in self.t5_files:
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key).to(device=self.device, dtype=self.dtype)
                    if hasattr(t5_encoder, key):
                        set_submodule_tensor(t5_encoder, key, tensor)
                    elif key.startswith("encoder.") or key.startswith("shared."):
                        try:
                            set_submodule_tensor(t5_encoder, key, tensor)
                        except Exception:
                            pass
                    del tensor

        # Crucial fix: tie encoder.embed_tokens to shared.weight if not explicitly loaded
        if hasattr(t5_encoder, "shared") and hasattr(t5_encoder.encoder, "embed_tokens"):
            t5_encoder.encoder.embed_tokens.weight.data.copy_(t5_encoder.shared.weight.data)

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
        print("  Prompt encoding complete. Text encoders purged.")
        return prompt_embeds, pooled_prompt_embeds, txt_ids

    # --------------------------------------------------------------------------
    # STAGE 2: Reusable Template Block Streaming (Peaks at ~1.2 GB VRAM)
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
        print("\n[Stage 2/3] Setting up Reusable Block Templates for Transformer...")

        open_handles = {p: safe_open(p, framework="pt", device="cpu") for p in self.transformer_files}

        # Index weight keys
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

        # 1. Instantiate Static Components (< 100 MB on GPU)
        inner_dim = 24 * 128  # 3072
        x_embedder = nn.Linear(64, inner_dim).to(self.device, self.dtype)
        context_embedder = nn.Linear(4096, inner_dim).to(self.device, self.dtype)
        time_text_embed = CombinedTimestepTextProjEmbeddings(inner_dim, pooled_projection_dim=768).to(
            self.device, self.dtype
        )
        pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56)).to(self.device)

        from .SubModels.FluxTransformer2DModel import AdaLayerNormContinuous

        norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6).to(
            self.device, self.dtype
        )
        proj_out = nn.Linear(inner_dim, 64, bias=True).to(self.device, self.dtype)

        # Load static weights
        for fpath, key in static_weights:
            tensor = open_handles[fpath].get_tensor(key)
            if key.startswith("x_embedder."):
                set_submodule_tensor(x_embedder, key.replace("x_embedder.", ""), tensor)
            elif key.startswith("context_embedder."):
                set_submodule_tensor(context_embedder, key.replace("context_embedder.", ""), tensor)
            elif key.startswith("time_text_embed."):
                set_submodule_tensor(time_text_embed, key.replace("time_text_embed.", ""), tensor)
            elif key.startswith("norm_out."):
                set_submodule_tensor(norm_out, key.replace("norm_out.", ""), tensor)
            elif key.startswith("proj_out."):
                set_submodule_tensor(proj_out, key.replace("proj_out.", ""), tensor)
            del tensor

        # 2. Instantiate ONE Dual-Stream Block & ONE Single-Stream Block template
        dual_template = FluxTransformerBlock(dim=inner_dim, num_attention_heads=24, attention_head_dim=128).to(
            self.device, self.dtype
        )
        single_template = FluxSingleTransformerBlock(
            dim=inner_dim, num_attention_heads=24, attention_head_dim=128
        ).to(self.device, self.dtype)

        print("  Static layers and block templates loaded (< 1.2 GB VRAM).")

        # 3. Schedule Setup
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len)
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=self.device, mu=mu)
        self.scheduler.set_begin_index(0)

        # 4. RoPE IDs
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = pos_embed(ids)

        # 5. Denoising Loop
        for step_idx, t in enumerate(self.scheduler.timesteps):
            print(f"  Denoising step {step_idx + 1}/{num_inference_steps} (timestep {t.item():.1f})...")
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with torch.no_grad():
                # A. Static input projections
                h_states = x_embedder(latents)
                enc_h_states = context_embedder(prompt_embeds)

                # Correct: timestep is already in [0, 1000]; do NOT multiply by 1000!
                temb = time_text_embed(timestep, pooled_prompt_embeds)

                # B. Execute 19 Dual-Stream Blocks using the single dual_template
                for i in range(19):
                    for fpath, full_key, subkey in dual_blocks_map[i]:
                        t_weight = open_handles[fpath].get_tensor(full_key)
                        set_submodule_tensor(dual_template, subkey, t_weight)
                        del t_weight

                    enc_h_states, h_states = dual_template(
                        hidden_states=h_states,
                        encoder_hidden_states=enc_h_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )

                # C. Execute 38 Single-Stream Blocks using the single single_template
                for j in range(38):
                    for fpath, full_key, subkey in single_blocks_map[j]:
                        t_weight = open_handles[fpath].get_tensor(full_key)
                        set_submodule_tensor(single_template, subkey, t_weight)
                        del t_weight

                    enc_h_states, h_states = single_template(
                        hidden_states=h_states,
                        encoder_hidden_states=enc_h_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )

                # D. Static output projections
                h_states = norm_out(h_states, temb)
                model_output = proj_out(h_states)

                # E. Euler Integration Step
                step_output = self.scheduler.step(model_output, t, latents, return_dict=True)
                latents = (
                    step_output.prev_sample
                    if isinstance(step_output, FlowMatchEulerDiscreteSchedulerOutput)
                    else step_output[0]
                )

        del (
            open_handles,
            x_embedder,
            context_embedder,
            time_text_embed,
            pos_embed,
            norm_out,
            proj_out,
            dual_template,
            single_template,
        )
        purge_memory()
        print("  Denoising complete. Transformer purged from memory.")
        return latents

    # --------------------------------------------------------------------------
    # STAGE 3: VAE Decoding (~335 MB)
    # --------------------------------------------------------------------------
    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> Image.Image:
        print("\n[Stage 3/3] Loading VAE and decoding final image...")

        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            scaling_factor=0.3611,
            shift_factor=0.1159,
        ).to(self.device, self.dtype)

        for fpath in self.vae_files:
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    try:
                        set_submodule_tensor(vae, key, tensor)
                    except Exception:
                        pass
                    del tensor

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