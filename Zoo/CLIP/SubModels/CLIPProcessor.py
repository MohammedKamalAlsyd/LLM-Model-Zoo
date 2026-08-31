import torch
from torchvision import transforms
from transformers import PreTrainedTokenizerFast

class CLIPProcessor:
    def __init__(self, tokenizer_id: str = "openai/clip-vit-base-patch32"):
        # We use HF just to grab the BPE Tokenizer rules to save 300 lines of regex code
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_id)
        
        # Standard CLIP normalizations exactly replicating HF's image processor
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def process_text(self, text: list[str], device: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        return tokens.input_ids.to(device)

    def process_image(self, images, device: str) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
            
        tensor_images = torch.stack([self.image_transform(img) for img in images])
        return tensor_images.to(device)