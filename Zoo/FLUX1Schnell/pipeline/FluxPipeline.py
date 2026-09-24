"""Low-VRAM Sequential Streaming Pipeline for FLUX.1 [schnell] text-to-image synthesis."""

import collections
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from PIL import Image
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from Zoo.Common.model_loader import get_safetensors_shards, purge_memory, set_submodule_tensor
from Zoo.Common.vision_utils import pack_latents_2d, prepare_multiaxis_coordinate_grid, unpack_latents_2d
from Zoo.FLUX1Schnell.configs import FluxConfig
from Zoo.FLUX1Schnell.modules.AutoEncoderKL import AutoencoderKL, DecoderOutput
from Zoo.FLUX1Schnell.modules.FluxTransformer2DModel import (
    AdaLayerNormContinuous,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from Zoo.FLUX1Schnell.modules.SchedulingFlowMatchEulerDiscrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from Zoo.FLUX1Schnell.modules.T5EncoderModel import T5EncoderModel


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculates sequence-length-dependent time shift mu for Flow Matching."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


class FluxPipeline:
    """Production streaming pipeline for FLUX.1 [schnell] capable of running on < 15 GB VRAM.

    Orchestrates generation through a 3-stage sequential lifecycle:
      Stage 1: Text conditioning via CLIP-L and T5-XXL (models purged after use).
      Stage 2: Sequential MMDiT execution using single reusable block templates.
      Stage 3: Autoencoder latent decoding to RGB (VAE purged after use).
    """

    def __init__(
        self,
        config: Optional[FluxConfig] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initializes the pipeline and resolves safetensors file handles.

        Args:
            config: Optional FluxConfig instance.
            device: Target execution device.
            dtype: Target compute precision (defaults to torch.bfloat16).
        """
        self.config = config or FluxConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.repo_id = self.config.repo_id
        self.vae_scale_factor = 8

        print(f"Indexing safetensors shards for '{self.repo_id}'...")
        self.transformer_files = get_safetensors_shards(self.repo_id, subfolder="transformer")
        self.t5_files = get_safetensors_shards(self.repo_id, subfolder="text_encoder_2")
        self.vae_files = get_safetensors_shards(self.repo_id, subfolder="vae")

        self.scheduler = FlowMatchEulerDiscreteScheduler(self.config.scheduler_config)

    # --------------------------------------------------------------------------
    # STAGE 1: Text Conditioning (Load -> Encode -> Purge)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _encode_prompt(
        self, prompt: str, max_sequence_length: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes text prompts using CLIP-L and T5-XXL, then purges both from memory."""
        print("[Stage 1/3] Loading Text Conditioning Models...")

        # 1. CLIP-L Pooled Projection (~250 MB)
        tokenizer = CLIPTokenizer.from_pretrained(self.repo_id, subfolder="tokenizer")
        clip_cls: Any = CLIPTextModel
        text_encoder = cast(
            nn.Module,
            clip_cls.from_pretrained(self.repo_id, subfolder="text_encoder", torch_dtype=self.dtype),
        ).to(self.device)

        clip_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_out = text_encoder(clip_inputs["input_ids"].to(self.device))
        pooled = getattr(clip_out, "pooler_output", clip_out[1])
        pooled_prompt_embeds = pooled.to(dtype=self.dtype, device=self.device)

        del text_encoder, tokenizer, clip_out
        purge_memory()

        # 2. T5-XXL Token Sequence (~9.5 GB)
        tokenizer_2 = AutoTokenizer.from_pretrained(self.repo_id, subfolder="tokenizer_2")
        t5_encoder = T5EncoderModel(self.config.t5_config).to(self.device, dtype=self.dtype)

        print("  Streaming T5-XXL shards directly into GPU...")
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

        t5_encoder.tie_weights()

        t5_inputs = tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_out = t5_encoder(input_ids=t5_inputs["input_ids"].to(self.device))
        prompt_embeds = t5_out.last_hidden_state.to(dtype=self.dtype, device=self.device)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        del t5_encoder, tokenizer_2, t5_out
        purge_memory()
        print("✓ Text conditioning complete. Text encoders purged.")
        return prompt_embeds, pooled_prompt_embeds, txt_ids

    # --------------------------------------------------------------------------
    # STAGE 2: Reusable Template Block Streaming (Peak < 1.2 GB VRAM)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        num_inference_steps: int = 4,
    ) -> torch.Tensor:
        """Executes Flow-Matching denoising using on-demand layer-wise weight swapping."""
        print("[Stage 2/3] Setting up Reusable MMDiT Block Templates (< 1.2 GB VRAM)...")

        open_handles = {p: safe_open(p, framework="pt", device="cpu") for p in self.transformer_files}

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

        # Instantiate static input/output projections (< 100 MB VRAM)
        inner_dim = 24 * 128  # 3072
        x_embedder = nn.Linear(64, inner_dim).to(self.device, self.dtype)
        context_embedder = nn.Linear(4096, inner_dim).to(self.device, self.dtype)
        time_text_embed = CombinedTimestepTextProjEmbeddings(inner_dim, pooled_projection_dim=768).to(
            self.device, self.dtype
        )
        pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56)).to(self.device)
        norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, eps=1e-6).to(self.device, self.dtype)
        proj_out = nn.Linear(inner_dim, 64, bias=True).to(self.device, self.dtype)

        # Populate static weights
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

        # Instantiate only ONE dual template and ONE single template
        dual_template = FluxTransformerBlock(dim=inner_dim, num_attention_heads=24, attention_head_dim=128).to(
            self.device, self.dtype
        )
        single_template = FluxSingleTransformerBlock(
            dim=inner_dim, num_attention_heads=24, attention_head_dim=128
        ).to(self.device, self.dtype)

        # Scheduler configuration and 3D RoPE coordinates
        mu = calculate_shift(latents.shape[1])
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=self.device, mu=mu)
        self.scheduler.set_begin_index(0)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = pos_embed(ids)

        # Denoising Euler Trajectory
        for step_idx, t in enumerate(self.scheduler.timesteps):
            print(f"  Step {step_idx + 1}/{num_inference_steps} (Timestep {t.item():.1f})...")
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            h_states = x_embedder(latents)
            enc_h_states = context_embedder(prompt_embeds)
            temb = time_text_embed(timestep, pooled_prompt_embeds)

            # Stream through 19 Dual-Stream blocks using dual_template
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

            # Stream through 38 Single-Stream blocks using single_template
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

            h_states = norm_out(h_states, temb)
            model_output = proj_out(h_states)

            step_out = self.scheduler.step(model_output, t, latents, return_dict=True)
            latents = (
                step_out.prev_sample
                if isinstance(step_out, FlowMatchEulerDiscreteSchedulerOutput)
                else step_out[0]
            )

        del open_handles, x_embedder, context_embedder, time_text_embed, pos_embed
        del norm_out, proj_out, dual_template, single_template
        purge_memory()
        print("✓ Latent trajectory denoised. Transformer templates purged.")
        return latents

    # --------------------------------------------------------------------------
    # STAGE 3: Autoencoder Decoding (Load -> Decode -> Purge)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> Image.Image:
        """Loads VAE, reconstructs RGB pixel image, and releases VAE parameters."""
        print("[Stage 3/3] Loading VAE and reconstructing pixels (~350 MB)...")

        vae = AutoencoderKL(self.config.vae_config).to(self.device, self.dtype)
        for fpath in self.vae_files:
            with safe_open(fpath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    try:
                        set_submodule_tensor(vae, key, tensor)
                    except Exception:
                        pass
                    del tensor

        latents = unpack_latents_2d(latents, height, width, self.vae_scale_factor, patch_size=2)
        latents = vae.unscale_latents(latents)

        dec_out = vae.decode(latents, return_dict=True)
        sample = dec_out.sample if isinstance(dec_out, DecoderOutput) else dec_out

        images = (sample / 2.0 + 0.5).clamp(0.0, 1.0)
        image_np = images.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        result_img = Image.fromarray((image_np * 255.0).round().astype("uint8"))

        del vae
        purge_memory()
        print("✓ Image rendered successfully. VAE purged.")
        return result_img

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
        seed: Optional[int] = None,
        max_sequence_length: int = 256,
    ) -> Image.Image:
        """Executes complete text-to-image Flow Matching synthesis."""
        height = 2 * (int(height) // 16) * 8
        width = 2 * (int(width) // 16) * 8

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        prompt_embeds, pooled_prompt_embeds, txt_ids = self._encode_prompt(
            prompt, max_sequence_length=max_sequence_length
        )

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        noise = torch.randn(
            (1, 16, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = pack_latents_2d(noise, patch_size=2)
        img_ids = prepare_multiaxis_coordinate_grid(height // 16, width // 16, device=self.device, dtype=self.dtype)

        latents = self._denoise_latents(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            num_inference_steps=num_inference_steps,
        )

        return self._decode_latents(latents, height, width)