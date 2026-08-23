import os
import time

import httpx
from dotenv import load_dotenv
from sarvamai import SarvamAI

from langfuse import observe

from utils.ffmpeg_utils import configure_pydub, get_ffmpeg_dir

load_dotenv()

# Sarvam REST API accepts up to 30s per request; use 25s for safety.
SARVAM_PIECE_SECONDS = 25
SARVAM_MAX_RETRIES = 4
SARVAM_TIMEOUT_SECONDS = 120

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

FFMPEG_DIR = get_ffmpeg_dir()
if FFMPEG_DIR:
    configure_pydub(FFMPEG_DIR)

_client: SarvamAI | None = None


def get_client() -> SarvamAI:
    global _client
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
    if _client is None:
        _client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    return _client


@observe(name="sarvam_stt")
def send_to_sarvam(piece_path: str, mode: str) -> str:
    """Send one <=30s audio file to Sarvam and return transcript text."""
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(1, SARVAM_MAX_RETRIES + 1):
        try:
            with open(piece_path, "rb") as audio_file:
                response = client.speech_to_text.transcribe(
                    file=audio_file,
                    model=SARVAM_STT_MODEL,
                    mode=mode,
                    request_options={
                        "timeout_in_seconds": SARVAM_TIMEOUT_SECONDS,
                        "max_retries": 2,
                    },
                )
            return (response.transcript or "").strip()
        except (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ) as exc:
            last_error = exc
            wait_s = min(2 ** attempt, 16)
            print(
                f"     network error on attempt {attempt}/{SARVAM_MAX_RETRIES}: {exc}. "
                f"Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)

    raise RuntimeError(
        f"Sarvam request failed after {SARVAM_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    """
    Transcribe one chunk with Sarvam Saaras.

    Language is auto-detected (English, Hindi, or other Indic).
    Long chunks are split into 25-second pieces, sent separately, then joined.
    """
    from pydub import AudioSegment

    mode = "translate" if translate else "transcribe"
    audio = AudioSegment.from_file(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = max(1, (len(audio) + piece_ms - 1) // piece_ms)

    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  -> Sarvam piece {i + 1}/{total_pieces} ({mode})...")
            full_text += send_to_sarvam(piece_path, mode=mode) + " "
            time.sleep(0.4)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


@observe(name="transcribe_all")
def transcribe_all(chunks: list[str], translate: bool = False) -> str:
    """Transcribe all audio chunks with Sarvam and return the full transcript."""
    mode_label = "translate to English" if translate else "transcribe"
    print(f"Using Sarvam ({SARVAM_STT_MODEL}, mode={mode_label}) for transcription.")

    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")   
        text = transcribe_chunk(chunk, translate=translate)
        full_transcript += text + " "

    full_transcript = full_transcript.strip()
    print("Transcription complete.")
    return full_transcript
