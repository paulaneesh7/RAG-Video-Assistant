from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.genai.summarize import summarize_transcript
from core.genai.summarize import generate_title
from core.genai.extractor import extract_action_items, extract_key_decision, extract_questions



if __name__ == "__main__":
    # source_file = "https://youtu.be/b2PESRl7De4?si=zGJRIL_18wpSjjKx"
    source_file = "https://youtu.be/EIFQWqTkPP4?si=A5l0EbtNvJhzUg2F"

    chunks = process_input(source_file)

    transcript = transcribe_all(chunks, translate=True)

    print(f"Transript: {transcript[:500] if len(transcript) > 500 else transcript}")

    summary = summarize_transcript(transcript)

    print(f"Summary: {summary}\n")
    print(f"Title: {generate_title(transcript)}\n")


    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decision(transcript)
    questions = extract_questions(transcript)

    print(f"Action Items: {action_items}\n")
    print(f"Key Decisions: {key_decisions}\n")
    print(f"Questions: {questions}\n")