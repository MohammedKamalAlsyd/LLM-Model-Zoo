# python -m Zoo.Chatterbox.serve
import os
import tempfile
import torch
import gradio as gr
from typing import Optional
from dotenv import load_dotenv

# Import the Pipeline we just built
from Zoo.Chatterbox.pipeline import ChatterboxPipeline

# Load environment variables (e.g., HF_TOKEN) if present
load_dotenv()

# Global pipeline instance with strict typing for Pylance
GLOBAL_PIPELINE: Optional[ChatterboxPipeline] = None

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
    "Arabic": "ar", "Hindi": "hi", "Russian": "ru", "Portuguese": "pt",
    "Dutch": "nl", "Turkish": "tr", "Polish": "pl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Greek": "el", "Hebrew": "he",
    "Malay": "ms", "Norwegian": "no", "Swahili": "sw"
}

def initialize_pipeline():
    """Initializes the model weights globally so they aren't reloaded on every request."""
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is None:
        print("Initializing Chatterbox V3 Multilingual Pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        GLOBAL_PIPELINE = ChatterboxPipeline(device=device)
        print("Pipeline successfully initialized!")


def gradio_predict(text_prompt: str, reference_audio_path: str, language: str, exaggeration: float, cfg_weight: float):
    """Wrapper function that connects the Gradio UI inputs to our pipeline."""
    assert GLOBAL_PIPELINE is not None, "Pipeline was not initialized."

    if not text_prompt.strip():
        raise gr.Error("Please enter some text for the AI to read.")
    if not reference_audio_path:
        raise gr.Error("Please upload or record a reference audio clip.")

    lang_code = SUPPORTED_LANGUAGES[language]
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "generated_speech.wav")

    try:
        GLOBAL_PIPELINE.generate_speech(
            text_prompt=text_prompt,
            reference_audio_path=reference_audio_path,
            language_id=lang_code,
            output_path=output_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {str(e)}")


def launch_app():
    # Load weights into VRAM before starting the server
    initialize_pipeline()
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    # Removed the theme argument to satisfy Pylance
    with gr.Blocks(title="Chatterbox V3 Multilingual TTS") as demo:
        gr.Markdown(f"# 🎙️ Chatterbox Voice Cloning (Multilingual V3)")
        gr.Markdown(f"Running on **{device_name}** | Built using LLaMA, Flow Matching, and HiFT-Net.")
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text Prompt", 
                    lines=4, 
                    placeholder="Enter the text you want the AI to speak here..."
                )
                
                lang_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()), 
                    value="English", 
                    label="Language"
                )
                
                ref_audio = gr.Audio(
                    label="Reference Voice (Upload or Record 3-5 seconds)", 
                    type="filepath", 
                    format="wav"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    exaggeration_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Emotion Exaggeration")
                    cfg_slider = gr.Slider(0.0, 5.0, value=1.5, step=0.1, label="CFG Weight (Adherence to Prompt)")

                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            # Right Column: Output
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated AI Speech", 
                    type="filepath", 
                    interactive=False
                )
                
        generate_btn.click(
            fn=gradio_predict,
            inputs=[text_input, ref_audio, lang_dropdown, exaggeration_slider, cfg_slider],
            outputs=[audio_output]
        )

    demo.launch(server_name="127.0.0.1", share=True, debug=True)

if __name__ == "__main__":
    launch_app()