"""Universal Speech-to-Text Studio for Open-Vocabulary Transcription & Translation.

Powered by a native clean-room PyTorch implementation of Whisper large-v3-turbo
with FlashAttention, KV-caching, and arbitrary-length audio chunk streaming.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
from transformers import WhisperTokenizer

# Setup repository paths
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent if (current_dir.parent / "Zoo").exists() else current_dir
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Zoo.Common.audio_utils import chunk_long_audio, format_timestamp, load_and_resample_audio
from Zoo.Common.model_loader import auto_detect_device_and_dtype, load_hf_model_weights, purge_memory
from Zoo.Whisper.configs import WhisperConfig
from Zoo.Whisper.processing.WhisperProcessing import WhisperFeatureExtractor
from Zoo.Whisper.Whisper import WhisperForConditionalGeneration

DEFAULT_WHISPER_REPO = "openai/whisper-large-v3-turbo"
DEVICE, DTYPE = auto_detect_device_and_dtype()

# --- Predefined Audio Examples ---
AUDIO_EXAMPLES = [
    [
        "https://upload.wikimedia.org/wikipedia/commons/7/75/Apollo_11_launch_clip.ogg",
        "en",
        "transcribe",
        True,
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Declaration_of_Human_Rights_-_Article_1_-_German.ogg",
        "de",
        "translate",
        False,
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/2/22/French_sample.ogg",
        "fr",
        "transcribe",
        True,
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/8/82/Albert_Einstein%27s_voice.ogg",
        "en",
        "transcribe",
        False,
    ],
]

LANGUAGE_CHOICES = [
    ("Auto-Detect", "auto"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Arabic", "ar"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Russian", "ru"),
    ("Hindi", "hi"),
]


def download_if_url(path_or_url: Optional[str]) -> Optional[str]:
    """Downloads remote audio files with request headers to prevent 403 Forbidden errors."""
    if not path_or_url or not isinstance(path_or_url, str):
        return None
    if path_or_url.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "whisper_cache")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(path_or_url.split("?")[0])
        local_filepath = os.path.join(cache_dir, filename)

        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) > 0:
            return local_filepath

        try:
            req = urllib.request.Request(
                path_or_url,
                headers={"User-Agent": "LLMModelZoo/1.0 AudioLoader"},
            )
            with urllib.request.urlopen(req, timeout=20) as response, open(local_filepath, "wb") as out_file:
                out_file.write(response.read())
            return local_filepath
        except Exception as e:
            print(f"Warning: Failed to fetch remote audio {path_or_url}: {e}")
            return None
    return path_or_url


class SpeechToTextManager:
    """Manages memory lifecycle and lazy-loading for Whisper speech backbones."""

    def __init__(self) -> None:
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperFeatureExtractor] = None
        self.tokenizer: Optional[WhisperTokenizer] = None
        self.active_repo: Optional[str] = None

    def unload(self) -> None:
        """Purges active model weights from system RAM and GPU VRAM."""
        if self.model is not None:
            print(f"[*] Unloading '{self.active_repo}'...")
            del self.model
            del self.processor
            del self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.active_repo = None
            purge_memory()

    def get_model(
        self, repo_id: str = DEFAULT_WHISPER_REPO
    ) -> Tuple[WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer]:
        """Loads weights using universal model_loader from Zoo.Common."""
        # Explicitly check all three to satisfy static typecheckers
        if (
            self.model is not None
            and self.processor is not None
            and self.tokenizer is not None
            and self.active_repo == repo_id
        ):
            return self.model, self.processor, self.tokenizer

        self.unload()
        print(f"[*] Initializing Whisper large-v3-turbo on {DEVICE.upper()} in {DTYPE}...")

        config = WhisperConfig.from_pretrained_name(repo_id)
        model = WhisperForConditionalGeneration(config)

        # Load weights using universal common loader
        load_hf_model_weights(
            model=model,
            repo_id=repo_id,
            allow_patterns=["*.safetensors", "*.bin"],
            strict=True,
            device=DEVICE,
            dtype=DTYPE,
        )

        processor = WhisperFeatureExtractor(config.audio_config)
        tokenizer = WhisperTokenizer.from_pretrained(repo_id)

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.active_repo = repo_id
        return model, processor, tokenizer


MANAGER = SpeechToTextManager()


@torch.no_grad()
def transcribe_speech(
    audio_path: Optional[str],
    language: str = "auto",
    task: str = "transcribe",
    with_timestamps: bool = True,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    """Transcribes or translates arbitrary-length audio files using long-form chunking and KV-caching."""
    if not audio_path:
        return "", "Please upload or record an audio file."

    resolved_path = download_if_url(audio_path)
    if not resolved_path or not os.path.exists(resolved_path):
        return "", "Could not resolve or access audio file."

    model, processor, tokenizer = MANAGER.get_model()
    model_dtype = next(model.parameters()).dtype

    # 1. Standardize and Resample to 16 kHz Mono
    waveform, sample_rate = load_and_resample_audio(resolved_path, target_sample_rate=16000)
    total_duration_sec = waveform.shape[-1] / sample_rate

    # 2. Slice audio into overlapping 30s segments with 25s stride
    chunks = list(chunk_long_audio(waveform, chunk_length_seconds=30.0, stride_seconds=25.0, sample_rate=16000))
    total_chunks = len(chunks)

    transcript_segments: List[str] = []

    # 3. Process sequential chunks
    for idx, (chunk_wave, start_t, end_t) in enumerate(chunks):
        progress((idx + 1) / total_chunks, desc=f"Processing chunk {idx + 1}/{total_chunks}...")

        # Extract 128-bin Mel Spectrogram: [1, 128, 3000]
        mel = processor(chunk_wave).to(device=DEVICE, dtype=model_dtype)

        # Target language handling (auto-detect falls back to "en" for prompt tokens)
        target_lang = "en" if language == "auto" else language

        # Run KV-cached autoregressive generation
        generated_token_ids = model.generate(
            input_features=mel,
            tokenizer=tokenizer,
            language=target_lang,
            task=task,
            without_timestamps=True,
            max_new_tokens=448,
        )

        chunk_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        if chunk_text:
            if with_timestamps:
                ts_header = f"[{format_timestamp(start_t)} --> {format_timestamp(end_t)}]"
                transcript_segments.append(f"{ts_header} {chunk_text}")
            else:
                transcript_segments.append(chunk_text)

    full_transcription = "\n".join(transcript_segments) if with_timestamps else " ".join(transcript_segments)
    summary = f"Completed {task} for {total_duration_sec:.2f}s of audio across {total_chunks} chunk(s)."
    return full_transcription, summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Whisper Speech Studio") as app:
        gr.Markdown("# 🎙️ Whisper Speech-to-Text Studio (`large-v3-turbo`)")
        gr.Markdown(
            f"Zero-Shot Speech Recognition & Translation running on **`{DEVICE.upper()}`** in **`{DTYPE}`**.<br/>"
            "Equipped with **KV-Caching** and **Arbitrary-Length Audio Streaming**."
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio or Record Voice")
                with gr.Row():
                    lang_select = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value="auto",
                        label="Source Language",
                    )
                    task_select = gr.Radio(
                        choices=[("Transcribe", "transcribe"), ("Translate to English", "translate")],
                        value="transcribe",
                        label="Task",
                    )
                timestamps_chk = gr.Checkbox(value=True, label="Generate Timestamp Segments")
                submit_btn = gr.Button("Transcribe Speech", variant="primary")

            with gr.Column():
                text_output = gr.Textbox(label="Transcription / Translation", lines=14)
                status_output = gr.Textbox(label="Processing Summary", lines=2)

        gr.Markdown("### 💡 Predefined Speech Examples")
        gr.Examples(
            examples=AUDIO_EXAMPLES,
            inputs=[audio_input, lang_select, task_select, timestamps_chk],
            outputs=[text_output, status_output],
            fn=transcribe_speech,
            cache_examples=False,
        )

        submit_btn.click(
            fn=transcribe_speech,
            inputs=[audio_input, lang_select, task_select, timestamps_chk],
            outputs=[text_output, status_output],
        )

    return app


if __name__ == "__main__":
    server = build_ui()
    server.launch(server_name="0.0.0.0", server_port=7863, share=True)