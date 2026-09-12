from dataclasses import dataclass, field


@dataclass
class Qwen3VLVisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    out_hidden_size: int = 2560
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [5, 11, 17])


@dataclass
class Qwen3VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5000000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


@dataclass
class Qwen3VLConfig:
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653