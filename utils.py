import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from pydub import AudioSegment
import io

# ================================
# 1. LOAD TRAINED MODEL (LoRA)
# ================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "openai/whisper-base"
LORA_PATH = "./whisper_lora_base"

print(f"--- Iniciando VozClara ---")
print(f"Dispositivo: {DEVICE}")
print(f"Carregando modelo base: '{BASE_MODEL_NAME}'...")

try:
    # Load tokenizer + feature extractor
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_NAME,
        language="portuguese",
        task="transcribe"
    )

    # Load base Whisper
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    # Apply LoRA adapter
    print(f"Aplicando fine-tuning da pasta: {LORA_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    model.to(DEVICE)
    print("✅ SUCESSO: Modelo com LoRA carregado!")

except Exception as e:
    print(f"❌ ERRO ao carregar o LoRA: {e}")
    print("⚠️  Usando modelo padrão da OpenAI como fallback...")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(DEVICE)


# ==================================================
# 2. PROCESS AUDIO + RUN WHISPER TRANSCRIPTION
# ==================================================

def process_wav_bytes(audio_file):
    """
    Recebe bytes de um WAV e retorna a transcrição final.
    """
    try:
        # Load WAV from bytes
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_file), format="wav")

        # Ensure mono + 16 kHz
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)

        # Convert to numpy
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

        # Normalize
        if audio_segment.sample_width == 2:
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:
            samples = samples / 2147483648.0
        else:
            samples = samples / 128.0

        # Silence warning
        if np.abs(samples).max() < 0.01:
            return "AVISO: Áudio muito baixo! Tente falar mais alto."

        # Prepare Whisper input features
        input_features = processor(
            samples,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # Generate transcription
        predicted_ids = model.generate(
            input_features,
            max_length=448,
            num_beams=5,
            do_sample=True,
            temperature=0.4,
        )

        # Decode into text
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    except Exception as e:
        print("Erro detalhado:", e)
        return f"Erro ao processar áudio: {e}"
