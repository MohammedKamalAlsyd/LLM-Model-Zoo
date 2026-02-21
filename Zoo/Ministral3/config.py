# Create Seperate Python File to Avoid Circular Imports between Mistral3 MultiModal Projector and Ministral3 Multimodal 

from dataclasses import dataclass, field
# Keep these imports here so the config can instantiate its defaults
from SubModels.Ministral3 import Ministral3Config
from SubModels.Pixtral import PixtralConfig

@dataclass
class Ministral3MultimodalConfig:
    spatial_merge_size: int = 2
    image_token_index: int = 10
    vision_feature_layer: int = -1
    tie_word_embeddings: bool = False
    text_config: Ministral3Config = field(default_factory=Ministral3Config)
    vision_config: PixtralConfig = field(default_factory=PixtralConfig)