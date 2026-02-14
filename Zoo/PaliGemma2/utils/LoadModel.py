from PaliGemma2 import PaliGemma2ForConditionalGeneration, PaliGemma2Config
from transformers import AutoTokenizer
import json
from safetensors import safe_open
from typing import Tuple, Union
from pathlib import Path
import logging
import torch

def load_hf_model(model_path: Union[str, Path], device: str = "cpu") -> Tuple[PaliGemma2ForConditionalGeneration, AutoTokenizer]:
    """Load a HuggingFace-style model directory saved with safetensors.

    Args:
        model_path: Path to the model directory containing `config.json` and one
            or more `*.safetensors` files.
        device: Torch device string to move the model to (default: "cpu").

    Returns:
        A tuple of (model, tokenizer).
    """
    logger = logging.getLogger(__name__)

    model_path = Path(model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.padding_side != "right":
        logger.warning("Loaded tokenizer.padding_side=%s (expected 'right')", tokenizer.padding_side)

    # Collect safetensors files
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        logger.warning("No .safetensors files found in %s", model_path)

    # Load tensors from safetensors into a dict
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load configuration
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with config_file.open("r", encoding="utf-8") as f:
        model_config = json.load(f)
    config = PaliGemma2Config(model_config)

    # Instantiate model and move to device
    model = PaliGemma2ForConditionalGeneration(config).to(device)

    # Move loaded tensors to the target device and load state dict
    if tensors:
        state_dict = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.info("Skipping state_dict load: no tensors available for %s", model_path)

    # Tie weights if model supports it
    try:
        model.tie_weights()
    except Exception:
        logger.debug("Model has no tie_weights() or it failed; continuing.")

    return model, tokenizer