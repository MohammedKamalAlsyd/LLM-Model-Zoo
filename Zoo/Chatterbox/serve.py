# python -m Zoo.Chatterbox.serve
import os
import tempfile
import torch
import gradio as gr
from typing import Optional
from dotenv import load_dotenv

# Import the unified pipeline
from Zoo.Chatterbox.pipeline import ChatterboxPipeline

# Load environment variables (e.g., HF_TOKEN)
load_dotenv()

# Global pipeline instance with strict typing for Pylance
GLOBAL_PIPELINE: Optional[ChatterboxPipeline] = None


def initialize_pipeline():
    """Initializes the model weights globally so they aren't reloaded on every request."""
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is None:
        print("Initializing Chatterbox TTS Pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        GLOBAL_PIPELINE = ChatterboxPipeline(device=device)
        print("Pipeline successfully initialized!")


def gradio_predict(text_prompt: str, reference_audio_path: str) -> str:
    """
    Wrapper function that connects the Gradio UI inputs to our pipeline.
    """
    # Satisfy Pylance strict type-checking
    assert GLOBAL_PIPELINE is not None, "Pipeline was not initialized."

    if not text_prompt.strip():
        raise gr.Error("Please enter some text for the AI to read.")
    if not reference_audio_path:
        raise gr.Error("Please upload or record a reference audio clip.")

    # Create a temporary directory to save the output wav file
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "generated_speech.wav")

    try:
        GLOBAL_PIPELINE.generate_speech(
            text_prompt=text_prompt,
            reference_audio_path=reference_audio_path,
            output_path=output_path
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {str(e)}")


def launch_app():
    # Load weights into VRAM before starting the server
    initialize_pipeline()
    
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    with gr.Blocks(title="Chatterbox Zero-Shot TTS") as demo:
        gr.Markdown("# 🎙️ Chatterbox Voice Cloning (From Scratch)")
        gr.Markdown(f"Running on **{device_name}** | Built using GPT-2, Flow Matching, and HiFT-Net.")
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text Prompt", 
                    lines=5, 
                    placeholder="Enter the text you want the AI to speak here...\n\nExample: 'Hello world, this is a test of my custom voice cloning pipeline!'"
                )
                
                ref_audio = gr.Audio(
                    label="Reference Voice (Upload or Record 3-5 seconds)", 
                    type="filepath", 
                    format="wav"
                )
                
                generate_btn = gr.Button("Generate Speech", variant="primary")
            
            # Right Column: Output
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated AI Speech", 
                    type="filepath", 
                    interactive=False
                )
                
        # Link the button to the prediction function
        generate_btn.click(
            fn=gradio_predict,
            inputs=[text_input, ref_audio],
            outputs=[audio_output]
        )

    # Launch web server locally
    demo.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    launch_app()