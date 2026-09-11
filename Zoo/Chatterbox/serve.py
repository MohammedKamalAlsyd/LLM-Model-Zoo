import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
import torch
import gradio as gr
from dotenv import load_dotenv

from Zoo.Chatterbox.pipeline import ChatterboxPipeline

load_dotenv()

# Windows-specific DLL path loading for FFmpeg
if os.name == "nt":
    FFMPEG_BIN = Path(r"C:\ffmpeg\bin")
    if FFMPEG_BIN.exists():
        os.add_dll_directory(str(FFMPEG_BIN))
        os.environ["PATH"] = f"{FFMPEG_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

GLOBAL_PIPELINE: Optional[ChatterboxPipeline] = None

# Supported Languages Mapping (Display Name -> Language Code)
SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
    "Arabic": "ar", "Hindi": "hi", "Russian": "ru", "Portuguese": "pt",
    "Dutch": "nl", "Turkish": "tr", "Polish": "pl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Greek": "el", "Hebrew": "he",
    "Malay": "ms", "Norwegian": "no", "Swahili": "sw"
}

# Official Chatterbox Multilingual V3 Reference Prompts & Transcripts
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een جديدة mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}


def download_if_url(audio_path: str) -> str:
    """Downloads remote audio URLs to local temporary file for librosa compatibility."""
    if audio_path.startswith(("http://", "https://")):
        cache_dir = os.path.join(tempfile.gettempdir(), "chatterbox_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_filename = os.path.join(cache_dir, os.path.basename(audio_path.split("?")[0]))
        if not os.path.exists(local_filename):
            print(f"Downloading reference sample: {audio_path}...")
            urllib.request.urlretrieve(audio_path, local_filename)
        return local_filename
    return audio_path


def initialize_pipeline():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is None:
        print("Initializing Chatterbox V3 Multilingual Pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        GLOBAL_PIPELINE = ChatterboxPipeline(device=device)
        print("Pipeline initialized successfully!")


def gradio_predict(
    text_prompt: str,
    reference_audio_path: str,
    language: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    seed: int
):
    assert GLOBAL_PIPELINE is not None, "Pipeline is not initialized."

    if not text_prompt.strip():
        raise gr.Error("Please enter text for the AI to speak.")
    if not reference_audio_path:
        raise gr.Error("Please provide or select a reference audio sample.")

    lang_code = SUPPORTED_LANGUAGES[language]
    resolved_audio_path = download_if_url(reference_audio_path)

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "generated_speech.wav")

    try:
        GLOBAL_PIPELINE.generate_speech(
            text_prompt=text_prompt,
            reference_audio_path=resolved_audio_path,
            language_id=lang_code,
            output_path=output_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            seed=int(seed) if seed > 0 else None,
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"Error during speech generation: {str(e)}")


def on_language_change(language_name: str):
    """Automatically swaps sample text and reference audio when target language changes."""
    lang_code = SUPPORTED_LANGUAGES.get(language_name, "en")
    cfg = LANGUAGE_CONFIG.get(lang_code, {})
    return cfg.get("text", ""), cfg.get("audio", None)


def launch_app():
    initialize_pipeline()
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    default_lang = "English"
    default_code = SUPPORTED_LANGUAGES[default_lang]
    default_sample = LANGUAGE_CONFIG[default_code]

    with gr.Blocks(title="Chatterbox V3 Multilingual TTS") as demo:
        gr.Markdown("# 🎙️ Chatterbox Voice Cloning (Multilingual V3)")
        gr.Markdown(
            f"Running on **{device_name}** | Powered by LLaMA-520M, Continuous Flow Matching, and HiFT-Net."
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text Prompt",
                    lines=4,
                    value=default_sample["text"],
                    placeholder="Enter the text you want the AI to speak here..."
                )
                
                lang_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value=default_lang,
                    label="Language"
                )
                
                ref_audio = gr.Audio(
                    label="Reference Voice (Upload, record, or use preset)",
                    type="filepath",
                    value=default_sample["audio"],
                    sources=["upload", "microphone"]
                )

                gr.Markdown(
                    "💡 **Note**: When cloning across different languages (e.g. reading English with a French voice), "
                    "set **CFG Weight to 0.0** to prevent accent bleeding."
                )

                with gr.Accordion("Advanced Settings", open=False):
                    exaggeration_slider = gr.Slider(
                        0.25, 2.0, value=0.5, step=0.05,
                        label="Emotion Exaggeration (Neutral = 0.5)"
                    )
                    cfg_slider = gr.Slider(
                        0.0, 2.0, value=0.5, step=0.05,
                        label="CFG Weight (Adherence to prompt; 0.0 for cross-lingual accent transfer)"
                    )
                    temp_slider = gr.Slider(
                        0.1, 1.5, value=0.8, step=0.05,
                        label="Temperature"
                    )
                    seed_input = gr.Number(
                        value=0,
                        label="Random Seed (0 for randomized generation)",
                        precision=0
                    )

                generate_btn = gr.Button("Generate Speech", variant="primary")

            # Right Column: Output
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated AI Speech",
                    type="filepath",
                    interactive=False
                )

        # Dynamic Language Switching Event
        lang_dropdown.change(
            fn=on_language_change,
            inputs=[lang_dropdown],
            outputs=[text_input, ref_audio],
            show_progress="hidden"
        )

        # Synthesis Trigger
        generate_btn.click(
            fn=gradio_predict,
            inputs=[
                text_input,
                ref_audio,
                lang_dropdown,
                exaggeration_slider,
                cfg_slider,
                temp_slider,
                seed_input
            ],
            outputs=[audio_output]
        )

    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    launch_app()