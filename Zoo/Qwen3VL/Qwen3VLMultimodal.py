import itertools
import torch
import torch.nn as nn

from .config import Qwen3VLConfig
from .SubModels.Text import Qwen3VLTextModel
from .SubModels.Vision import Qwen3VLVisionModel
from .utils.KVCache import KVCache


class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3VLConfig | None = None):
        super().__init__()
        self.config = config or Qwen3VLConfig()

        vc = self.config.vision_config
        tc = self.config.text_config

        self.visual = Qwen3VLVisionModel(
            depth=vc.depth,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size,
            num_heads=vc.num_heads,
            out_hidden_size=vc.out_hidden_size,
            patch_size=vc.patch_size,
            temporal_patch_size=vc.temporal_patch_size,
            spatial_merge_size=vc.spatial_merge_size,
            num_position_embeddings=vc.num_position_embeddings,
            deepstack_visual_indexes=vc.deepstack_visual_indexes,
        )

        self.language_model = Qwen3VLTextModel(
            vocab_size=tc.vocab_size,
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            num_layers=tc.num_hidden_layers,
            num_heads=tc.num_attention_heads,
            num_kv_heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,
            rope_theta=tc.rope_theta,
            mrope_section=tc.mrope_section,
            eps=tc.rms_norm_eps,
        )

        self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)
        if tc.tie_word_embeddings:
            self.lm_head.weight = self.language_model.embed_tokens.weight

        self.rope_deltas: torch.Tensor | None = None

    def get_vision_position_ids(self, start_pos: int, grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Constructs 3D coordinates [T, H, W] for vision tokens."""
        sms = int(self.config.vision_config.spatial_merge_size)
        t = int(grid_thw[0].item())
        h = int(grid_thw[1].item()) // sms
        w = int(grid_thw[2].item()) // sms

        pos_t = torch.zeros(t, device=device, dtype=torch.long)
        pos_h = torch.arange(h, device=device, dtype=torch.long) + start_pos
        pos_w = torch.arange(w, device=device, dtype=torch.long) + start_pos

        T, H, W = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
        return torch.stack([T, H, W], dim=0).reshape(3, -1) + start_pos

    def build_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates 3D position IDs for M-RoPE across text and visual spans."""
        bs, seq_len = input_ids.shape
        device = input_ids.device
        sms = int(self.config.vision_config.spatial_merge_size)
        pos_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)
        rope_deltas: list[int] = []

        img_iter = iter(image_grid_thw) if image_grid_thw is not None else None

        for b in range(bs):
            token_types = mm_token_type_ids[b].tolist()
            groups = [
                (modality, len(list(group)))
                for modality, group in itertools.groupby(token_types)
            ]
            cur_pos: int = 0
            b_positions: list[torch.Tensor] = []
            for modality, length in groups:
                if modality == 0:  # Text
                    b_positions.append(
                        torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + cur_pos
                    )
                    cur_pos += length
                else:  # Image / Video
                    if img_iter is None:
                        raise ValueError("image_grid_thw must be provided when multimodal tokens exist.")
                    grid = next(img_iter)
                    v_pos = self.get_vision_position_ids(cur_pos, grid, device=device)
                    b_positions.append(v_pos)
                    step = int(max(int(grid[1].item()), int(grid[2].item()))) // sms
                    cur_pos += step

            all_pos = torch.cat(b_positions, dim=1)
            pos_ids[:, b] = all_pos
            rope_deltas.append(int(all_pos.max().item()) + 1 - seq_len)

        deltas = torch.tensor(rope_deltas, device=device).unsqueeze(1)
        return pos_ids, deltas

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        visual_pos_mask = None
        deepstack_features = None

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds, deepstack_features = self.visual(pixel_values, image_grid_thw)
            image_mask = input_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype))
            visual_pos_mask = image_mask

        # Generate M-RoPE 3D positions if not provided
        if position_ids is None:
            if mm_token_type_ids is not None and image_grid_thw is not None:
                position_ids, self.rope_deltas = self.build_3d_position_ids(
                    input_ids, mm_token_type_ids, image_grid_thw
                )
            else:
                seq_len = input_ids.shape[1]
                past = kv_cache.get_seq_length() if kv_cache else 0
                pos = torch.arange(past, past + seq_len, device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                if self.rope_deltas is not None:
                    pos = pos + self.rope_deltas
                position_ids = pos

        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            kv_cache=kv_cache,
            visual_pos_mask=visual_pos_mask,
            deepstack_embeds=deepstack_features,
        )

        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        eos_token_id: int = 151645,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        kv_cache = KVCache(device=str(input_ids.device), dtype=self.lm_head.weight.dtype)

        # Prefill step
        logits = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            kv_cache=kv_cache,
        )
        kv_cache.advance(input_ids.shape[1])

        cur_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = [cur_token.item()]

        for _ in range(max_new_tokens - 1):
            if cur_token.item() == eos_token_id:
                break

            logits = self.forward(input_ids=cur_token, kv_cache=kv_cache)
            kv_cache.advance(1)

            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float("Inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            cur_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(cur_token.item())
            yield cur_token.item()