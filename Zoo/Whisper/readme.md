# Whisper (`large-v3-turbo`) — Clean-Room PyTorch Implementation

A native, dependency-light PyTorch implementation of OpenAI's **Whisper large-v3-turbo** for open-vocabulary automatic speech recognition (ASR), zero-shot translation, and arbitrary-length audio transcription.

---

## 🧠 Architectural Overview

Whisper is an encoder-decoder Transformer that processes audio via **Log-Mel Spectrograms** and decodes text tokens autoregressively.

```mermaid
graph TD
    Audio[16kHz Raw Waveform] --> Mel[128-bin Log-Mel Spectrogram: B x 128 x 3000]
    Mel --> Conv[Strided 1D Convolutions: 3000 -> 1500 Frames]
    Conv --> Enc[32-Layer Audio Encoder: 1280 dim]
    Enc --> CrossAttn[Cross-Attention (Audio Keys & Values)]
    Prompt[Target Language & Task Tokens] --> Dec[4-Layer Turbo Decoder + KV-Cache]
    CrossAttn --> Dec
    Dec --> LMHead[Tied LM Head: 51,866 Vocab]
    LMHead --> Text[Transcription / Translation Output]
```

- **Detailed Tensor Flow**: See [`assets/detailed_diagram.mmd`](./assets/detailed_diagram.mmd)

---

## ⚙️ Key Technical Features

1. **Turbo Architecture (4-Layer Decoder)**:
   While standard `large-v3` uses 32 decoder layers, `large-v3-turbo` achieves an **$8\times$ speedup** by condensing the text decoder into **4 layers** while retaining the full **32-layer audio encoder**.
2. **Stateful Key-Value (KV) Caching ($O(N)$ Decoding)**:
   Eliminates the quadratic $O(N^2)$ decoding bottleneck by caching historical self-attention and static cross-attention matrices.
3. **Arbitrary-Length Audio Streaming**:
   Processes audio of any length via a sliding-window chunking engine (`chunk_long_audio`) with a 30.0s context window and 25.0s stride, preventing truncation.
4. **Universal Shard Loading**:
   Uses `Zoo.Common.model_loader.load_hf_model_weights` for RAM-safe shard streaming with 100% `strict=True` parameter alignment against official Hugging Face checkpoints (`openai/whisper-large-v3-turbo`).
5. **99+ Languages & Zero-Shot Translation**:
   Supports multilingual transcription and direct English translation via dynamic prompt token injection.

---

## 📐 Mathematical Formulation

### 1. Log-Mel Spectrogram Transformation

Given a 16 kHz discrete waveform $x[n]$:
$$\text{STFT}(x)[k, m] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$
where window length $N = 400$ (25ms), hop length $H = 160$ (10ms), and $w[n]$ is a Hann window. The power spectrum is projected onto 128 Slaney mel filterbanks $M$:
$$S_{\text{mel}} = M \cdot |\text{STFT}(x)|^2, \quad S_{\text{log}} = \log_{10}(\max(S_{\text{mel}}, 10^{-10}))$$
Normalized via dynamic range compression:
$$S_{\text{norm}} = \frac{\max(S_{\text{log}}, \max(S_{\text{log}}) - 8.0) + 4.0}{4.0}$$

### 2. Temporal Downsampling

The spectrogram $S_{\text{norm}} \in \mathbb{R}^{B \times 128 \times 3000}$ passes through two 1D convolutions:

- $\text{Conv1D}(128, 1280, k=3, s=1)$
- $\text{Conv1D}(1280, 1280, k=3, s=2)$
  Shrinking temporal frames from $3000 \to 1500$, matching the 20ms frame resolution of speech.

---

## 🚀 Quickstart

Run the dedicated Gradio Speech-to-Text Studio:

```bash
python Zoo/serve_speech_to_text.py
```
