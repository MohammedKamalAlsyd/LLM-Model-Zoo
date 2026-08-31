import torch
from torch import nn
from Zoo.CLIP.SubModels.CLIPTextModel import CLIPTextModel
from Zoo.CLIP.SubModels.CLIPVisionModel import CLIPVisionModel

class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel()
        self.vision_model = CLIPVisionModel()

        # Projection layers to map both models to 512 dimensions
        self.visual_projection = nn.Linear(768, 512, bias=False)
        self.text_projection = nn.Linear(512, 512, bias=False)
        
        # Learned temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor):
        # 1. Get Summary Tokens
        vision_pooled = self.vision_model(pixel_values)
        text_pooled = self.text_model(input_ids)

        # 2. Project into shared 512-D Space
        image_embeds = self.visual_projection(vision_pooled)
        text_embeds = self.text_projection(text_pooled)

        # 3. L2 Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # 4. Cosine Similarity (Scores)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text