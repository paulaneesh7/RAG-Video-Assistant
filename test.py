from utils.audio_processor import process_input
from core.transcriber import transcribe_all



if __name__ == "__main__":
    # source_file = "https://youtu.be/b2PESRl7De4?si=zGJRIL_18wpSjjKx"
    source_file = "https://youtu.be/EIFQWqTkPP4?si=A5l0EbtNvJhzUg2F"

    chunks = process_input(source_file)

    transcript = transcribe_all(chunks, translate=True)

    print(transcript)