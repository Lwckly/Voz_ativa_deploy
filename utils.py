import speech_recognition as sr
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pydub import AudioSegment
import io


# ---------------------------------------------------------------------
# 🎙️ 1️⃣ AUDIO CAPTURE + PROCESSING FUNCTION
# ---------------------------------------------------------------------
def audio_record():
    """
    Records audio from the microphone, processes and normalizes it,
    returning a NumPy array, logs, and status_record dictionary.
    """
    logs = []
    status_record = {
        "mic_ready": 0,
        "audio_recorded": 0,
        "audio_processed": 0,
        "record_success": 0
    }

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(sample_rate=16000) as source:
            logs.append("🎙️ Ajustando para o ruído ambiente...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            status_record["mic_ready"] = 1

            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            logs.append("✅ Gravação concluída.")
            status_record["audio_recorded"] = 1

            wav_bytes = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")

            logs.append(
                f"Duração: {len(audio_segment)/1000:.2f}s | "
                f"Sample rate: {audio_segment.frame_rate}Hz | "
                f"Canais: {audio_segment.channels}"
            )

            # Convert to mono and ensure 16kHz
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            if audio_segment.frame_rate != 16000:
                audio_segment = audio_segment.set_frame_rate(16000)

            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

            # Normalize audio
            if audio_segment.sample_width == 2:
                samples = samples / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples / 2147483648.0
            else:
                samples = samples / 128.0

            logs.append(
                f"✅ Áudio processado: shape={samples.shape}, "
                f"min/max={samples.min():.4f}/{samples.max():.4f}"
            )
            status_record["audio_processed"] = 1
            status_record["record_success"] = 1

            return data, logs, status_record

    except Exception as e:
        logs.append(f"❌ Erro na captura ou processamento do áudio: {e}")
        return None, logs, status_record


# ---------------------------------------------------------------------
# 🧠 2️⃣ WHISPER MODEL FUNCTION
# ---------------------------------------------------------------------
def audio_transcribe(data):
    """
    Loads Whisper model, processes the provided audio samples, and
    returns the transcription, logs, and status_transcribe dictionary.
    """
    logs = []
    status_transcribe = {
        "model_loaded": 0,
        "features_extracted": 0,
        "transcription_generated": 0,
        "transcribe_success": 0
        "device":0
    }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "openai/whisper-base"

    logs.append(f"🧠 Usando dispositivo: {DEVICE}")
    if device == "cuda":
        status_transcribe["device"]=1
    logs.append(f"Carregando modelo Whisper: '{MODEL_NAME}'...")

    try:
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        logs.append("✅ Modelo Whisper carregado com sucesso!")
        status_transcribe["model_loaded"] = 1
    except Exception as e:
        logs.append(f"❌ Erro ao carregar modelo: {e}")
        return None, logs, status_transcribe

    try:
        input_features = processor(
            samples,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)
        logs.append(f"🎛️ Features extraídas: shape={input_features.shape}")
        status_transcribe["features_extracted"] = 1

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
        logs.append(f"✅ Transcrição gerada: '{transcription}'")
        status_transcribe["transcription_generated"] = 1
        status_transcribe["transcribe_success"] = 1

        return transcription, logs, status_transcribe

    except Exception as e:
        logs.append(f"❌ Erro durante a transcrição: {e}")
        return None, logs, status_transcribe
