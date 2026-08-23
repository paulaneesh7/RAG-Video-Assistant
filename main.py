from dotenv import load_dotenv
from langfuse import observe

from core.genai.extractor import (
    extract_action_items,
    extract_key_decision,
    extract_questions,
)
from core.genai.rag_engine import ask_question, build_rag_chain
from core.genai.summarize import generate_title, summarize_transcript
from core.observability.langfuse import flush_langfuse, get_langfuse
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

load_dotenv()


@observe(name="run_pipeline")
def run_pipeline(source: str, translate: bool = False) -> dict:
    print("Starting AI Video Assistant Pipeline...")

    get_langfuse().update_current_span(
        metadata={"source": source, "translate": translate}
    )

    chunks = process_input(source)
    transcript = transcribe_all(chunks, translate=translate)
    print(f"Raw Transcription (first 300 characters): {transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize_transcript(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decision(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "transcript": transcript,
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # source_file = "https://www.youtube.com/watch?v=JKDjT_8LWzE"
    source_file = "https://www.youtube.com/watch?v=VgQiOYWr-1M"

    try:
        result = run_pipeline(source_file, translate=True)

        print("\n" + "=" * 60)
        print(f"📌 Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n✅ Action Items:\n{result['action_items']}")
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print("=" * 60)

        print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
        rag_chain = result["rag_chain"]
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
    finally:
        flush_langfuse()
