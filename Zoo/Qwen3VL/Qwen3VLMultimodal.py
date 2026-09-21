"""Master Multimodal Wrapper for Qwen3-VL.

Unifies Dynamic-Resolution Vision with DeepStack, 3D M-RoPE coordinate scheduling,
and the Qwen3 language model backbone with 1:1 Hugging Face weight parity.
"""

import itertools
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from Zoo.Common.KV_Cache import KVCache
from Zoo.Common.vision_utils import replace_image_tokens
from Zoo.Qwen3VL.configs import Qwen3VLConfig
from Zoo.Qwen3VL.modules.Text import Qwen3VLTextModel
from Zoo.Qwen3VL.modules.Vision import Qwen3VLVisionModel


class Qwen3VLForConditionalGeneration(nn.Module):
    """Top-level conditional generation model for Qwen3-VL."""

    def __init__(self, config: Optional[Qwen3VLConfig] = None) -> None:
        super().__init__()
        self.config = config or Qwen3VLConfig()

        self.visual = Qwen3VLVisionModel(self.config.vision_config)
        self.language_model = Qwen3VLTextModel(self.config.text_config)
        self.lm_head = nn.Linear(
            self.config.text_config.hidden_size,
            self.config.text_config.vocab_size,
            bias=False,
        )

        if self.config.text_config.tie_word_embeddings:
            self.tie_weights()

        self.rope_deltas: Optional[torch.Tensor] = None

    def tie_weights(self) -> None:
        """Ties LM head weights to word embeddings."""
        self.lm_head.weight = self.language_model.embed_tokens.weight

    def get_vision_position_ids(
        self,
        start_pos: int,
        grid_thw: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Constructs 3D [T, H, W] coordinate grids for visual tokens."""
        sms = int(self.config.vision_config.spatial_merge_size)
        t = int(grid_thw[0].item())
        h = int(grid_thw[1].item()) // sms
        w = int(grid_thw[2].item()) // sms

        pos_t = torch.zeros(t, device=device, dtype=torch.long)
        pos_h = torch.arange(h, device=device, dtype=torch.long) + start_pos
        pos_w = torch.arange(w, device=device, dtype=torch.long) + start_pos

        t_grid, h_grid, w_grid = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
        return torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1) + start_pos

    def build_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds 3D M-RoPE position coordinates across mixed text and image sequences."""
        bs, seq_len = input_ids.shape
        device = input_ids.device
        sms = int(self.config.vision_config.spatial_merge_size)
        pos_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)
        rope_deltas: List[int] = []

        img_iter = iter(image_grid_thw) if image_grid_thw is not None else None

        for b in range(bs):
            token_types = mm_token_type_ids[b].tolist()
            groups = [(modality, len(list(group))) for modality, group in itertools.groupby(token_types)]
            cur_pos: int = 0
            b_positions: List[torch.Tensor] = []

            for modality, length in groups:
                if modality == 0:  # Text Span
                    text_pos = torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + cur_pos
                    b_positions.append(text_pos)
                    cur_pos += length
                else:  # Image / Video Span
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
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        logits_to_keep: Union[int, slice] = 0,
    ) -> Dict[str, torch.Tensor]:
        """Conditional multimodal forward pass with DeepStack injection.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            pixel_values: Optional flattened patch pixels for visual inputs.
            image_grid_thw: Optional tensor holding [T, H, W] per visual input.
            mm_token_type_ids: Optional modality map (0=text, 1=multimodal).
            position_ids: Optional pre-constructed 3D position ID coordinates.
            kv_cache: Key-value cache instance for autoregressive decode.
            logits_to_keep: Number of trailing logit steps to compute (saves memory).

        Returns:
            Dictionary containing {"logits": (batch, seq_len, vocab_size)}.
        """
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        visual_pos_mask = None
        deepstack_features = None

        # 1. Process Vision and Inject Tokens
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds, deepstack_features = self.visual(pixel_values, image_grid_thw)
            inputs_embeds = replace_image_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
                image_token_id=self.config.image_token_id,
            )
            visual_pos_mask = (input_ids == self.config.image_token_id)

        # 2. Derive 3D M-RoPE Positions
        if position_ids is None:
            if mm_token_type_ids is not None and image_grid_thw is not None:
                position_ids, self.rope_deltas = self.build_3d_position_ids(
                    input_ids, mm_token_type_ids, image_grid_thw
                )
            else:
                seq_len = input_ids.shape[1]
                past = kv_cache.num_items() if kv_cache is not None else 0
                pos = torch.arange(past, past + seq_len, device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                if self.rope_deltas is not None:
                    pos = pos + self.rope_deltas
                position_ids = pos

        # 3. Language Backbone Pass
        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            kv_cache=kv_cache,
            visual_pos_mask=visual_pos_mask,
            deepstack_embeds=deepstack_features,
        )

        # Prefill memory optimization: compute logits only for requested trailing steps
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, slice):
            hidden_states = hidden_states[:, logits_to_keep, :]

        return {"logits": self.lm_head(hidden_states)}