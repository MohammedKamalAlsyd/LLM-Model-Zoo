from PaliGemma2 import PaliGemma2ForConditionalGeneration, PaliGemma2Config
from transformers import AutoTokenizer
import json
from safetensors import safe_open
from typing import Tuple, Union
from pathlib import Path
import logging
import torch
import gc


def load_hf_model(
    model_path: Union[str, Path],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[PaliGemma2ForConditionalGeneration, AutoTokenizer]:
    """Load a HuggingFace-style model directory efficiently.

    Loads weights incrementally to avoid RAM spikes.
    """
    logger = logging.getLogger(__name__)
    model_path = Path(model_path)

    # 1. Load Config
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with config_file.open("r", encoding="utf-8") as f:
        model_config = json.load(f)
    config = PaliGemma2Config(model_config)

    # 2. Initialize Model (Empty)
    print(f"Initializing model structure on {device}...")
    with torch.device("cuda" if device == "cuda" else "cpu"):
        model = PaliGemma2ForConditionalGeneration(config)

    # Force garbage collection to clear any initialization debris
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    # 4. Stream Weights from Safetensors
    safetensors_files = list(model_path.glob("*.safetensors"))
    print(f"Found weight files: {[f.name for f in safetensors_files]}")

    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors found in {model_path}")

    print("Streaming weights directly to GPU...")

    # Get the state dict keys of the model to verify mapping
    model_state_dict = model.state_dict()

    for sf in safetensors_files:
        print(f"Processing {sf.name}...")
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in model_state_dict:
                    # 1. Load tensor to CPU (small chunk)
                    tensor = f.get_tensor(key)

                    # 2. Move to GPU and cast to dtype immediately
                    tensor = tensor.to(device=device, dtype=dtype)

                    # 3. Update the model parameter in-place
                    # We have to navigate the module tree to set the data
                    _set_tensor_in_model(model, key, tensor)

                    # 4. Delete temp reference
                    del tensor
                else:
                    pass
                    # logger.warning(f"Key {key} from file not found in model definition.")

        # Cleanup after every file
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Tie weights manually if needed
    try:
        model.tie_weights()
    except Exception:
        pass

    return model, tokenizer


def _set_tensor_in_model(model: torch.nn.Module, key: str, tensor: torch.Tensor):
    """Helper to set a parameter tensor deep inside the model hierarchy."""
    try:
        # Split key "model.layers.0.weight" -> parent "model.layers.0", child "weight"
        if "." in key:
            module_name, param_name = key.rsplit(".", 1)
            # Retrieve the submodule (e.g., model.layers.0)
            submodule = model.get_submodule(module_name)
        else:
            submodule = model
            param_name = key

        # Get the current parameter
        current_param = getattr(submodule, param_name)

        if current_param.shape != tensor.shape:
            # Handle squeeze/unsqueeze differences if necessary (common in some checkpoints)
            if current_param.numel() == tensor.numel():
                tensor = tensor.view(current_param.shape)
            else:
                print(
                    f"Shape mismatch for {key}: Model {current_param.shape} vs Loaded {tensor.shape}"
                )
                return

        # Assign the data directly to the existing parameter on GPU
        # We use .data to overwrite the tensor content without tracking gradients
        with torch.no_grad():
            current_param.data = tensor

    except AttributeError:
        print(f"Could not set parameter {key}")
    except Exception as e:
        print(f"Error setting {key}: {e}")
