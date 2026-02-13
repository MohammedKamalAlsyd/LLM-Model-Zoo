# PaliGemma 2 - PyTorch Implementation

A modular, from-scratch implementation of the **PaliGemma 2** Vision-Language Model (VLM). This project integrates the **SigLip** vision encoder with the **Gemma 2** language model to perform tasks like image captioning and visual question answering (VQA).

## 🧠 High-Level Architecture

PaliGemma 2 follows a standard multimodal architecture: it "sees" by encoding images into patches, projects those patches into the language model's embedding space, and then "speaks" by generating text based on the combined visual and textual context.

![PaliGemma 2 Architecture](Zoo/PaliGemma2/assets/HigeViewModelArchitecture.png)

## 🔄 Logic Flow & Inference Pipeline

The following chart illustrates how data flows from the user input through the sub-models to produce a final text prediction.

```mermaid
graph TD
    %% Input Stage
    User([User Input]) -->|Raw Image| P_Img[Processor: Resize & Normalize]
    User -->|Text Prompt| P_Txt[Processor: Tokenize & Add Special Tokens]

    %% Vision Encoding Stage
    subgraph Vision_Tower [SigLip Vision Encoder]
        direction TB
        P_Img -->|Pixel Values (3, 224, 224)| SigLip[SigLip Transformer]
        SigLip -->|Patch Embeddings| SigLip_Out[Vision Embeddings]
    end

    %% Projection Stage
    subgraph Connection [Multi-Modal Projector]
        SigLip_Out -->|Linear Layer| Projector[Project to 2304 dim]
        Projector -->|Visual Tokens| Concat
    end

    %% Language Modeling Stage
    subgraph Language_Model [Gemma 2 Decoder]
        P_Txt -->|Input IDs| Embed[Text Embeddings]

        Concat(Concatenation) -->|Merged: <Image Tokens> + <Text Tokens>| Decoder[Gemma 2 Layers]
        Embed --> Concat

        Decoder -->|Hidden States| LM_Head[LM Head]
        LM_Head -->|Logits| Sampler{Greedy / Sampling}

        Sampler -->|Next Token| Output_Buf[Generated Token Buffer]

        %% Auto-regressive Loop
        Output_Buf -.->|Update Cache| KV_Cache[KV Cache]
        KV_Cache -.-> Decoder
    end

    %% Output Stage
    Output_Buf -->|End of Sequence?| Detokenizer[Detokenizer]
    Detokenizer --> Final([Final Text Prediction])

    style Vision_Tower fill:#e1f5fe,stroke:#01579b
    style Language_Model fill:#e8f5e9,stroke:#1b5e20
    style Connection fill:#fff3e0,stroke:#e65100
```

## 📂 Project Structure

The project is organized to separate the sub-models (Vision vs Text) from the logic that wires them together.

```text
LLM Model Zoo/
├── serve.py                  # 🚀 Main Entry Point: Gradio UI, Model Loading, & Inference Loop
├── saved_model/              # 💾 Local cache for downloaded/processed weights
└── Zoo/
    └── PaliGemma2/
        ├── assets/
        │   └── HigeViewModelArchitecture.png  # Architecture Diagram
        ├── config.json       # ⚙️ Hyperparameters (Vocab size, Layer counts, Dims)
        ├── PaliGemma2.py     # 🧠 Main Class: Wires Vision + Projector + Text
        ├── SubModels/
        │   ├── Gemma2.py             # Text Model (Attention, RoPE, RMSNorm)
        │   ├── SigLip.py             # Vision Model (Patch Embeddings, Encoder)
        │   └── PaliGemmaProjector.py # Linear Connector (Vision -> Text dim)
        └── utils/
            ├── KVCache.py            # ⚡ Optimization: Key-Value Cache for fast decoding
            └── PaliGemma2Processor.py# 🖼️ Preprocessing: Image Resizing & Text Tokenization
```

## 🛠️ Components Explained

### 1. SigLip.py (Vision)

- **Input**: Raw image tensor `(Batch, 3, 224, 224)`.
- **Action**: Breaks image into 14x14 patches, adds positional embeddings, and passes them through Transformer layers.
- **Output**: A sequence of feature vectors representing the image contents.

### 2. PaliGemmaProjector.py (Connector)

- **Action**: A simple Linear layer.
- **Purpose**: The Vision tower outputs dimension `1152`, but Gemma 2 expects dimension `2304`. This layer bridges that gap.

### 3. Gemma2.py (Text)

- **Input**: Concatenated sequence of `[Projected Image Features + Text Token Embeddings]`.
- **Action**: Uses Causal Masking, Rotary Positional Embeddings (RoPE), and Grouped Query Attention (GQA).
- **Output**: Probability distribution over the vocabulary for the **next** token.

### 4. KVCache.py (Optimization)

- **Purpose**: During text generation, we generate one word at a time. Instead of re-calculating the attention for all previous words every step, we cache the Keys and Values to speed up inference significantly.

## 🚀 How to Run

### Initialize Environment (using `uv`)

```powershell
uv init
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add gradio huggingface_hub transformers safetensors pillow numpy
```

### Start the Server

```powershell
uv run serve.py
```

- **First Run**: Weights will be downloaded from Hugging Face automatically.
- **Subsequent Runs**: Weights load from `saved_model/`.

### Access UI

Open `http://localhost:7860` in your browser.
