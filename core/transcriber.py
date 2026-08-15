import whisper
import os



WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

model = None


# Load the Whisper model
def load_whisper_model():
    global model

    if model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model loaded successfully.")
    return model