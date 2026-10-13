# 🦁 LLM Model Zoo: Clean-Room Multimodal & Generative Foundation Models

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10+ (CUDA 13.0)](https://img.shields.io/badge/PyTorch-2.10%20%7C%20CUDA%2013.0-EE4C2C.svg?logo=pytorch)](https://download.pytorch.org/whl/cu130)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?logo=astral)](https://astral.sh/uv)
[![Code Architecture: Clean--Room](https://img.shields.io/badge/Architecture-Clean--Room%20PyTorch-green.svg)](#-architectural-matrix)

A comprehensive, dependency-light, clean-room PyTorch implementation of state-of-the-art foundation models spanning **Large Language Models (LLMs)**, **Vision-Language Models (VLMs)**, **Flow-Matching & Diffusion Engines**, **3D Video Synthesis**, **Visual Grounding**, and **Speech/Audio AI**.

Built entirely from **first-principles PyTorch tensor algebra** with strict 1:1 checkpoint parameter parity against official Hugging Face checkpoints—eliminating opaque library wrappers.

---

## 🧠 Architectural Matrix

| Category             | Architecture              | Backbone / Engine                   | Highlights                                                                                                                    | Official Checkpoint                           |
| :------------------- | :------------------------ | :---------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| **VLM / Reasoning**  | **Qwen3-VL (4B)**         | Conv3D + 3D M-RoPE LM               | Spatio-temporal 3D patch convolution, DeepStack multi-layer visual feature injection, per-head RMSNorm                        | `Qwen/Qwen3-VL-4B-Instruct`                   |
| **VLM / Reasoning**  | **Ministral-3 (3B)**      | Pixtral Vision + Ministral-3 LM     | Native aspect ratio slicing, 2D continuous axial RoPE, 262k-token YaRN context scaling, 2x2 patch merger                      | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| **VLM / Grounding**  | **PaliGemma 2 (3B)**      | SigLIP + Gemma 2 + UViM VAE         | 2D bicubic patch interpolation, Sandwich RMSNorm, dual logit softcapping, discrete token mask decoder                         | `google/paligemma2-3b-mix-224`                |
| **Image Synthesis**  | **FLUX.1 [schnell]**      | 12B Dual/Single MMDiT               | Rectified Flow Matching ODE, sequence-dependent dynamic time shifting, streaming execution on < 1.2 GB VRAM                   | `black-forest-labs/FLUX.1-schnell`            |
| **Image Synthesis**  | **Stable Diffusion v1.5** | Latent UNet 2D + AutoencoderKL      | CompVis parameter parity, Tweedie-formulated DDPM ancestral sampling, cross-attention conditioning                            | `runwayml/stable-diffusion-v1-5`              |
| **Video & Foley**    | **Wan 2.1 (1.3B / 14B)**  | 3D Causal DiT + 3D Causal VAE       | Factorized 3D continuous polar RoPE (T, H, W), FlowUniPC multistep ODE solver, FoleyCrafter V2A audio co-generation           | `Wan-AI/Wan2.1-T2V-1.3B`                      |
| **Visual Grounding** | **SAM3**                  | ViT-H + FPN + Multimodal DETR       | Relative Position Bias (RPB), 200 learnable queries, RoI-Align box geometry, continuous video temporal memory tracking        | `facebook/sam3`                               |
| **Metric Alignment** | **CLIP**                  | ViT-B/32 + Causal Text Tower        | Dual-tower metric contrastive alignment, FlashAttention SDPA, QuickGELU non-linearities                                       | `openai/clip-vit-base-patch32`                |
| **Voice Cloning**    | **Chatterbox V3**         | LLaMA-520M (T3) + S3Flow + HiFT-Net | Autoregressive speech token prediction, Classifier-Free Guidance (CFG), continuous flow-matching mel generation, NSF vocoding | `ResembleAI/chatterbox`                       |
| **Speech-to-Text**   | **Whisper Turbo**         | 32L Enc + 4L Turbo Dec              | 128-bin Slaney log-mel extraction, O(1) KV-caching, arbitrary-length sliding window chunking                                  | `openai/whisper-large-v3-turbo`               |

---

## ⚡ Engineering & Performance Highlights

- **Zero-Leakage Shard Streaming (`Zoo.Common.model_loader`)**: Streams `.safetensors` files shard-by-shard directly into model memory structures with global key verification (`strict=True`), bypassing system RAM saturation.
- **Low-VRAM Sequential Block Swapping**: Generates 1024×1024 images with the 12-billion parameter FLUX MMDiT and T5-XXL on **< 1.2 GB VRAM** by maintaining single reusable block templates in GPU memory and reading layer weights on demand.
- **FlashAttention / Native SDPA**: Standardized across every attention layer (`F.scaled_dot_product_attention`), automatically utilizing FlashAttention-2 or memory-efficient kernels based on GPU hardware.
- **Modern Geometric Coordinates**: Native implementations of **3D Multimodal RoPE** (Qwen3-VL), **Factorized 3D Continuous RoPE** (Wan 2.1), **2D Continuous Axial RoPE** (Pixtral), and **Relative Position Bias** (SAM3).
- **High-Performance Audio Pipeline**: Full causal neural vocoding, 40Hz Butterworth filtering, cosine edge windowing, and peak mastering to -1.0 dBFS for generated audio and video.

---

## 📦 Installation & Setup

This repository is managed with [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package manager and resolver.

### 1. Prerequisites

- **Python `>= 3.13`**
- **NVIDIA Driver with CUDA 13.0+ support** (for Linux and Windows)
- **FFmpeg** installed and accessible in your system `PATH` (required for video rendering and audio muxing)

### 2. Install `uv`

If you don't have `uv` installed, install it using the official standalone installer:

```bash
# On Linux and macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Clone and Synchronize the Environment

Clone the repository and run `uv sync`. `uv` will automatically read `pyproject.toml`, configure the explicit PyTorch CUDA 13.0 index, create a dedicated `.venv`, and install all dependencies:

```bash
git clone https://github.com/<your-username>/llm-model-zoo.git
cd llm-model-zoo

# Synchronize virtual environment with CUDA 13.0 PyTorch wheels
uv sync
```

### 4. Configure Hugging Face Authentication (Optional)

Gated models (such as `mistralai/Ministral-3-3B-Instruct-2512-BF16`) require a Hugging Face User Access Token:

```bash
# On Linux/macOS:
export HF_TOKEN="your_huggingface_token_here"

# On Windows (PowerShell):
$env:HF_TOKEN="your_huggingface_token_here"
```

---

## 🚀 Interactive Web Studios (Gradio)

Launch any of the dedicated interactive web interfaces via `uv run`:

| Studio                      | Command                                       | Description                                                                                                        |
| :-------------------------- | :-------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **Multimodal VLM Studio**   | `uv run python -m Zoo.serve_multimodal`       | Unified evaluation of **Qwen3-VL**, **Ministral-3**, and **PaliGemma 2** with instance segmentation mask decoding. |
| **Unified Image Studio**    | `uv run python -m Zoo.generate_image`         | Interactive synthesis studio dynamically swapping between **FLUX.1 [schnell]** and **Stable Diffusion v1.5**.      |
| **Video & Foley Studio**    | `uv run python -m Zoo.server_create_video`    | Continuous Flow video generation via **Wan 2.1** with motion-synchronized **FoleyCrafter** audio co-synthesis.     |
| **Visual Grounding Studio** | `uv run python -m Zoo.serve_visual_grounding` | Open-vocabulary object grounding, dense semantic segmentation heatmaps, and temporal video tracking via **SAM3**.  |
| **Voice Cloning Studio**    | `uv run python -m Zoo.serve_voice_clone`      | 23-language zero-shot voice cloning with emotion control and accent transfer via **Chatterbox V3**.                |
| **Speech-to-Text Studio**   | `uv run python -m Zoo.serve_speech_to_text`   | Long-form transcription and translation using **Whisper large-v3-turbo** with KV-caching.                          |
| **Zero-Shot Studio**        | `uv run python -m Zoo.serve_zero_shot`        | Real-time contrastive image classification and metric distribution analysis via **CLIP**.                          |

---

## 💻 Programmatic Python API

### 1. Ultra Low-VRAM Image Generation (FLUX.1 [schnell])

```python
from Zoo.FLUX1Schnell.FLUX import FluxModel

# Initializes the streaming pipeline (keeps VRAM < 1.2 GB)
pipeline = FluxModel.from_pretrained_weights()

image = pipeline(
    prompt="A futuristic DeLorean cruising on a luminous neon highway in space, 8k resolution",
    height=1024,
    width=1024,
    num_inference_steps=4,
    seed=42,
)
image.save("flux_output.png")
```

### 2. Video Generation with Synchronized Audio (Wan 2.1)

```python
from Zoo.Wan21.Wan21 import Wan21

# Load pipeline with automatic layer offloading to CPU RAM
wan = Wan21.from_pretrained(
    task="t2v",
    scale="1.3B",
    resolution="480P",
    offload_model=True,
)

# Generates 33 frames of video alongside motion-synchronized Foley audio
video_path = wan.generate(
    prompt="A cinematic drone shot of a vintage convertible cruising along the Pacific coast highway at sunset.",
    steps=20,
    guide_scale=5.0,
    enable_audio=True,
)
print(f"Video with sound generated at: {video_path}")
```

### 3. Multimodal Reasoning & Grounding (Qwen3-VL)

```python
from PIL import Image
import torch
from transformers import AutoProcessor
from Zoo.Qwen3VL.Qwen3VLMultimodal import Qwen3VLForConditionalGeneration
from Zoo.Common.model_loader import load_hf_model_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen3VLForConditionalGeneration().to(device=device, dtype=torch.bfloat16)
load_hf_model_weights(model, repo_id="Qwen/Qwen3-VL-4B-Instruct", device=device, dtype=torch.bfloat16)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
image = Image.open("tests/assets/demo.jpg")

messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this scene in detail."}]}]
chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[chat_prompt], images=[image], return_tensors="pt").to(device)

output = model(**inputs, logits_to_keep=1)
# Autoregressive generation steps continue via model.language_model ...
```

### 4. Zero-Shot Multilingual Voice Cloning (Chatterbox V3)

```python
import torchaudio
from Zoo.Chatterbox.Chatterbox import ChatterboxModel

model, processor = ChatterboxModel.from_pretrained(device="cuda")

inputs = processor(
    text="Welcome to the clean-room foundation model zoo.",
    reference_audio="path/to/reference_sample.wav",
    language="en",
    device=model.device,
)

waveform = model.generate(
    text_tokens=inputs["text_tokens"],
    ref_24k=inputs["ref_24k"],
    ref_16k=inputs["ref_16k"],
    exaggeration=0.5,
    cfg_weight=0.5,
)

torchaudio.save("cloned_speech.wav", waveform, model.config.sample_rate_24k)
```

### 5. High-Speed Speech Transcription (Whisper large-v3-turbo)

```python
import torch
from transformers import WhisperTokenizer
from Zoo.Whisper.Whisper import WhisperForConditionalGeneration
from Zoo.Whisper.processing.WhisperProcessing import WhisperFeatureExtractor
from Zoo.Common.audio_utils import load_and_resample_audio
from Zoo.Common.model_loader import load_hf_model_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration().to(device=device, dtype=torch.float16)
load_hf_model_weights(model, repo_id="openai/whisper-large-v3-turbo", device=device, dtype=torch.float16)

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo")
extractor = WhisperFeatureExtractor(model.config.audio_config)

waveform, sr = load_and_resample_audio("path/to/speech.wav", target_sample_rate=16000)
mel = extractor(waveform).to(device=device, dtype=torch.float16)

tokens = model.generate(
    input_features=mel,
    tokenizer=tokenizer,
    language="en",
    task="transcribe",
)
text = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"Transcription: {text}")
```

---

## 📂 Repository Structure

```
LLM Model Zoo/
└── Zoo/
    ├── Chatterbox/          # TTS & Voice Cloning (T3 LLaMA, S3Flow, S3Gen, HiFT-Net Vocoder)
    ├── CLIP/                # Contrastive Language-Image Pre-Training (ViT + Causal Text)
    ├── Common/              # Universal Shared Infrastructure (RoPE, KV-Cache, Model Loader, Activations)
    │   ├── activations.py
    │   ├── attention_utils.py
    │   ├── audio_utils.py
    │   ├── KV_Cache.py
    │   ├── model_loader.py
    │   ├── multimodality_utils.py
    │   ├── RoPE.py
    │   └── vision_utils.py
    ├── FLUX1Schnell/        # 12B MMDiT Rectified Flow Text-to-Image Pipeline
    ├── Ministral3/          # Dynamic Aspect Ratio VLM (Pixtral Vision + Ministral-3 LM)
    ├── PaliGemma2/          # Vision-Language & Segmentation (SigLIP + Gemma 2 + UViM VAE)
    ├── Qwen3VL/             # Spatio-Temporal VLM (Conv3D, DeepStack, 3D M-RoPE)
    ├── SAM3/                # Segment Anything 3 (ViT Backbone, Multimodal DETR, Video Tracker)
    ├── StableDiffusion/     # Latent Diffusion v1.5 (UNet 2D, AutoencoderKL, DDPM)
    ├── Wan21/               # 3D Video Foundation Model (DiT, Causal VAE, FoleyCrafter V2A)
    ├── Whisper/             # Speech Foundation Model (128-bin Mel, Turbo Decoder, O(1) KV-Cache)
    ├── generate_image.py    # Multi-model Image Generation Studio
    ├── serve_multimodal.py  # Unified VLM Studio (Qwen3-VL, Ministral-3, PaliGemma 2)
    ├── serve_speech_to_text.py
    ├── serve_visual_grounding.py
    ├── serve_voice_clone.py
    ├── serve_zero_shot.py
    └── server_create_video.py
```

---

## 📄 License & Attribution

This project is released under the **Apache 2.0 License**.

Pretrained weights are distributed under their respective upstream open licenses:

- `FLUX.1 [schnell]`: Apache 2.0
- `Qwen3-VL`: Apache 2.0
- `Wan 2.1`: Apache 2.0
- `Whisper`: MIT License
- `SAM3`: Creative Commons Non-Commercial (or respective repository distribution license)
- `PaliGemma 2` & `Gemma 2`: Gemma Terms of Use
- `Ministral-3`: Mistral Research / Commercial License