"""Unified Zero-Shot Multilingual Voice Cloning Studio (Chatterbox V3).

Features dynamic lazy loading, lifecycle memory management, URL reference caching,
and multi-dialect parameter controls.
"""

import gc
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple
import urllib.request
import gradio as gr
import numpy as np
import torch
import torchaudio

# Repository path resolution
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent if (current_dir.parent / "Zoo").exists() else current_dir
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Zoo.Chatterbox.Chatterbox import ChatterboxModel
from Zoo.Chatterbox.processing.ChatterboxProcessor import ChatterboxProcessor
from Zoo.Common.model_loader import auto_detect_device_and_dtype

DEVICE, DTYPE = auto_detect_device_and_dtype()

# Windows DLL loading for FFmpeg if present
if os.name == "nt":
    ffmpeg_bin = Path(r"C:\ffmpeg\bin")
    if ffmpeg_bin.exists():
        os.add_dll_directory(str(ffmpeg_bin))
        os.environ["PATH"] = f"{ffmpeg_bin}{os.pathsep}{os.environ.get('PATH', '')}"

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
    "Arabic": "ar", "Hindi": "hi", "Russian": "ru", "Portuguese": "pt",
    "Dutch": "nl", "Turkish": "tr", "Polish": "pl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Greek": "el", "Hebrew": "he",
    "Malay": "ms", "Norwegian": "no", "Swahili": "sw",
}

LANGUAGE_PRESETS: Dict[str, Dict[str, str]] = {
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel.",
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube.",
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube.",
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal.",
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。",
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。",
    },
}


def download_if_url(audio_path: Optional[str]) -> Optional[str]:
    """Downloads remote reference audio with headers preventing 403 Forbidden responses."""
    if not audio_path or not isinstance(audio_path, str):
        return None
    if audio_path.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "modelzoo_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_filename = os.path.join(cache_dir, os.path.basename(audio_path.split("?")[0]))
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
            return local_filename
        try:
            req = urllib.request.Request(
                audio_path,
                headers={"User-Agent": "LLMModelZoo/1.0 Python-urllib"},
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(local_filename, "wb") as out_file:
                out_file.write(response.read())
            return local_filename
        except Exception as e:
            print(f"Warning: Could not download audio {audio_path}: {e}")
            return None
    return audio_path


class ChatterboxManager:
    """Manages memory lifecycle and lazy initialization for Chatterbox."""

    def __init__(self) -> None:
        self.model: Optional[ChatterboxModel] = None
        self.processor: Optional[ChatterboxProcessor] = None

    def unload(self) -> None:
        """Purges active model weights to free VRAM."""
        if self.model is not None:
            print("[*] Unloading Chatterbox to free VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_model(self) -> Tuple[ChatterboxModel, ChatterboxProcessor]:
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        print(f"[*] Initializing Chatterbox V3 on {DEVICE.upper()} in {DTYPE}...")
        model, processor = ChatterboxModel.from_pretrained(device=DEVICE)
        self.model = model
        self.processor = processor
        return model, processor


MANAGER = ChatterboxManager()


def run_synthesis(
    text: str,
    reference_audio: Any,
    language_name: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    seed: int,
) -> str:
    """Synthesis endpoint processing text and reference voice into synthesized WAV."""
    if not text or not text.strip():
        raise gr.Error("Please enter text for the model to speak.")
    if reference_audio is None:
        raise gr.Error("Please upload or select a reference voice audio sample.")

    lang_code = SUPPORTED_LANGUAGES.get(language_name, "en")
    resolved_audio_path = download_if_url(reference_audio)
    if resolved_audio_path is None or not os.path.exists(resolved_audio_path):
        raise gr.Error("Failed to locate or download reference audio sample.")

    model, processor = MANAGER.get_model()

    batch = processor(
        text=text,
        reference_audio=resolved_audio_path,
        language=lang_code,
        device=model.device,
    )

    wav = model.generate(
        text_tokens=batch["text_tokens"],
        ref_24k=batch["ref_24k"],
        ref_16k=batch["ref_16k"],
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        seed=int(seed) if seed > 0 else None,
    )

    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "synthesized_voice.wav")
    torchaudio.save(out_path, wav, model.config.sample_rate_24k)
    return out_path


def build_ui() -> gr.Blocks:
    default_lang = "English"
    default_code = SUPPORTED_LANGUAGES[default_lang]
    default_preset = LANGUAGE_PRESETS.get(default_code, LANGUAGE_PRESETS["en"])
    initial_audio_path = download_if_url(default_preset["audio"])

    with gr.Blocks(title="Chatterbox Voice Cloning Studio") as app:
        gr.Markdown("# 🎙️ Chatterbox Zero-Shot Voice Studio")
        gr.Markdown(
            f"Running on **`{DEVICE.upper()}`** in **`{DTYPE}`**.<br/>"
            "High-fidelity multilingual cloning driven by LLaMA-520M, Continuous Flow Matching, and HiFT-Net."
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Text Prompt",
                    lines=4,
                    value=default_preset["text"],
                    placeholder="Enter speech prompt here...",
                )

                lang_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value=default_lang,
                    label="Target Language",
                )

                ref_audio_input = gr.Audio(
                    label="Reference Voice Sample",
                    type="filepath",
                    value=initial_audio_path,
                    sources=["upload", "microphone"],
                )

                with gr.Accordion("Generation Parameters", open=False):
                    exaggeration_slider = gr.Slider(
                        0.2, 1.8, value=0.5, step=0.05, label="Emotion Intensity"
                    )
                    cfg_slider = gr.Slider(
                        0.0, 2.0, value=0.5, step=0.05,
                        label="CFG Weight (Set to 0.0 for cross-lingual accent transfer)"
                    )
                    temp_slider = gr.Slider(
                        0.2, 1.4, value=0.8, step=0.05, label="Sampling Temperature"
                    )
                    seed_input = gr.Number(
                        value=0, label="Random Seed (0 for randomized synthesis)", precision=0
                    )

                synthesize_btn = gr.Button("Synthesize Speech", variant="primary")

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Synthesized Audio Output",
                    type="filepath",
                    interactive=False,
                )

        def on_language_change(lang_name: str):
            code = SUPPORTED_LANGUAGES.get(lang_name, "en")
            preset = LANGUAGE_PRESETS.get(code, {})
            resolved_audio = download_if_url(preset.get("audio"))
            return preset.get("text", ""), resolved_audio

        lang_dropdown.change(
            fn=on_language_change,
            inputs=[lang_dropdown],
            outputs=[text_input, ref_audio_input],
        )

        synthesize_btn.click(
            fn=run_synthesis,
            inputs=[
                text_input,
                ref_audio_input,
                lang_dropdown,
                exaggeration_slider,
                cfg_slider,
                temp_slider,
                seed_input,
            ],
            outputs=[output_audio],
        )

    return app


if __name__ == "__main__":
    server = build_ui()
    server.launch(server_name="0.0.0.0", server_port=7862, share=True)