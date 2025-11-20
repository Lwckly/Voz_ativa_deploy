import speech_recognition as sr
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pydub import AudioSegment
import io
import librosa

# --- 1. CARREGAMENTO DO MODELO WHISPER ---
# Esta parte é a mesma do nosso servidor, carregamos o modelo uma vez.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/whisper-base"

#print("--- Iniciando Script de Teste de Transcrição ---")
#print(f"Usando dispositivo: {DEVICE}")
#print(f"Carregando modelo Whisper: '{MODEL_NAME}'...")

try:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro fatal ao carregar o modelo: {e}")
    exit()

# --- 2. LÓGICA DE GRAVAÇÃO E TRANSCRIÇÃO ---

def process_wav_bytes(audio_file):
    """
    Processa um áudio WAV bruto (bytes) e retorna a transcrição usando Whisper.

    Args:
        wav_bytes (bytes): Bytes do arquivo WAV (16-bit PCM, mono, 16kHz).

    Returns:
        str: Transcrição do áudio.
    """
    try:
        # Carrega áudio usando pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_file), format="wav")

        # Debug
       # print(f"Duração do áudio: {len(audio_segment) / 1000:.2f} segundos")
        #print(f"Sample rate original: {audio_segment.frame_rate} Hz")
        #print(f"Canais: {audio_segment.channels}")
       # print(f"Sample width: {audio_segment.sample_width} bytes")

        # Mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Sample rate 16kHz
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)

        # Bytes → NumPy
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

        # Normalização
        if audio_segment.sample_width == 2:      # PCM16
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:    # PCM32
            samples = samples / 2147483648.0
        else:
            samples = samples / 128.0            # PCM8

        #print(f"Shape do array de áudio: {samples.shape}")
       # print(f"Valores min/max do áudio: {samples.min():.4f} / {samples.max():.4f}")

        if np.abs(samples).max() < 0.01:
            return (f"AVISO: Áudio muito baixo! Tente falar mais alto.")
        else:
            pass
        # Whisper processing
       # print("Processando com Whisper...")

        input_features = processor(
            samples,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

       # print(f"Shape das features de entrada: {input_features.shape}")

        predicted_ids = model.generate(
            input_features,
            max_length=448,
            num_beams=5,
            do_sample=True,
            temperature=0.6,
            language="pt",
            task="transcribe"
        )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        #print("-" * 50)
        #print(f"TEXTO TRANSCRITO: '{transcription}'")
        #print(f"Tamanho da transcrição: {len(transcription)} caracteres")
        #print("-" * 50)

        return transcription

    except sr.UnknownValueError:
        return ("Não foi possível entender o áudio. Tente falar mais claramente.")
    except sr.RequestError as e:
        return (f"Erro no serviço de reconhecimento; {e}")
    except Exception as e:
        return (f"Ocorreu um erro inesperado: {e}")
