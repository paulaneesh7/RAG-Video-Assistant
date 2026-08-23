import os

import yt_dlp

from langfuse import observe

from utils.ffmpeg_utils import configure_pydub, get_ffmpeg_dir

FFMPEG_DIR = get_ffmpeg_dir()
if FFMPEG_DIR:
    configure_pydub(FFMPEG_DIR)

DOWNLOAD_PATH = "downloads"
os.makedirs(DOWNLOAD_PATH, exist_ok=True)


# Helper function to configure yt-dlp for YouTube audio downloads
def youtube_ydl_opts(output_path: str) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": FFMPEG_DIR,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }


# Function to download audio from YouTube using yt-dlp
def download_audio_from_youtube(url: str) -> str:
    if not FFMPEG_DIR:
        raise RuntimeError(
            "FFmpeg not found. Install it (winget install Gyan.FFmpeg) or set FFMPEG_PATH "
            "to the folder containing ffmpeg.exe."
        )

    output_path = os.path.join(DOWNLOAD_PATH, "%(title)s.%(ext)s")
    with yt_dlp.YoutubeDL(youtube_ydl_opts(output_path)) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        return base + ".wav"



# Function to convert any audio/video file to WAV format using pydub
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    from pydub import AudioSegment

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz mono audio
    audio.export(output_path, format="wav")
    return output_path



# Function to chunk audio into 10 minute chunks
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks



@observe(name="process_input")
def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected Youtube URL. Downloading audio...")
        wav_path = download_audio_from_youtube(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Chunked audio into - {len(chunks)} chuns(s) created successfully.")
    return chunks






if __name__ == "__main__":
    data = download_audio_from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(convert_to_wav(data))
