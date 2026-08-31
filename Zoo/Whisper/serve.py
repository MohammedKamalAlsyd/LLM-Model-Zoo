import torch
import torch.nn as nn
import gradio as gr
import soundfile as sf
import torchaudio.transforms as T
from transformers import WhisperTokenizer
from Zoo.Whisper.SubModels.AudioProcessing import WhisperFeatureExtractor
from Zoo.Whisper.utils.model_loader import load_whisper_weights

# ====================================================================
# Global State & Initialization
# ====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device.upper()}")

# Load our from-scratch PyTorch models
ENCODER, DECODER = load_whisper_weights(device)
ENCODER.eval()
DECODER.eval()

# Load our Audio Signal Processor (No weights)
AUDIO_PROCESSOR = WhisperFeatureExtractor(feature_size=128)

# Load the Tokenizer
TOKENIZER = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo")

# LM Head
LM_HEAD = nn.Linear(1280, 51866, bias=False).to(device)
LM_HEAD.weight = DECODER.embed_tokens.weight


# ====================================================================
# The Core Generation Pipeline
# ====================================================================
def transcribe(audio_path: str) -> str:
    if audio_path is None:
        return "Please upload or record an audio file."

    with torch.no_grad():
        # 1. Read Audio with soundfile
        audio_data, sample_rate = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(audio_data)

        # Handle tensor shapes
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, Seq_Len)
        else:
            waveform = waveform.transpose(0, 1)  # (Channels, Seq_Len)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        mel_spectrogram = AUDIO_PROCESSOR(waveform).to(device)

        # 2. Run the Audio Encoder
        encoder_hidden_states = ENCODER(mel_spectrogram)

        # 3. Setup the Prompt Tokens Dynamically (Fixes the Token ID shift bug!)
        prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        prompt_ids = TOKENIZER.convert_tokens_to_ids(prompt_tokens)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 4. The Autoregressive Generation Loop
        max_tokens = 448
        generated_tokens = []
        eos_token_id = TOKENIZER.convert_tokens_to_ids("<|endoftext|>")

        print("Generating transcription...")
        for _ in range(max_tokens):
            decoder_outputs = DECODER(input_ids, encoder_hidden_states)
            
            # Last token projection
            last_token_hidden_state = decoder_outputs[:, -1, :]
            logits = LM_HEAD(last_token_hidden_state)
            
            # Greedy Decoding
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
            
            # Append next token
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            active_token = next_token_id.item()
            generated_tokens.append(active_token)
            
            # Stop if EOS token is predicted
            if active_token == eos_token_id:
                break

        # 5. Decode the tokens into text
        transcription = TOKENIZER.decode(generated_tokens, skip_special_tokens=True)
        return transcription.strip()


# ====================================================================
# Gradio UI Setup
# ====================================================================
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="PyTorch Whisper large-v3-turbo (From Scratch)",
    description="A clean, native PyTorch implementation of OpenAI's Whisper model processing audio via Log-Mel Spectrograms and generating text autoregressively.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)