import os

from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

load_dotenv()

PROMPT_LABEL = "production"

PROMPT_NAMES = {
    "summarize_map_system": "summarize_map_system_prompt",
    "summarize_map_user": "summarize_map_user_prompt",
    "summarize_reduce_system": "summarize_reduce_system_prompt",
    "summarize_reduce_user": "summarize_reduce_user_prompt",
    "meeting_title_system": "meeting_title_system_prompt",
    "meeting_title_user": "meeting_title_user_prompt",
    "action_items_system": "action_items_system_prompt",
    "key_decision_system": "key_decision_system_prompt",
    "questions_system": "questions_system_prompt",
    "rag_system": "rag_system_prompt",
    "rag_user": "rag_user_prompt",
}

__all__ = [
    "PROMPT_NAMES",
    "observe",
    "get_langfuse",
    "get_langfuse_handler",
    "flush_langfuse",
    "get_text_prompt",
    "get_chat_prompt_from_pair",
    "chain_config",
]


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


_client: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=_env("LANGFUSE_PUBLIC_KEY"),
            secret_key=_env("LANGFUSE_SECRET_KEY"),
            host=_env("LANGFUSE_HOST") or _env("LANGFUSE_BASE_URL"),
        )
    return _client


def get_langfuse_handler() -> CallbackHandler:
    return CallbackHandler(public_key=_env("LANGFUSE_PUBLIC_KEY"))


def flush_langfuse() -> None:
    get_langfuse().flush()


def get_text_prompt(name: str, *, label: str = PROMPT_LABEL):
    client = get_langfuse()
    try:
        return client.get_prompt(name, label=label, type="text")
    except Exception:
        return client.get_prompt(name, label="latest", type="text")


def get_chat_prompt_from_pair(system_name: str, user_name: str, *, label: str = PROMPT_LABEL):
    system_prompt = get_text_prompt(system_name, label=label)
    user_prompt = get_text_prompt(user_name, label=label)
    return system_prompt, user_prompt


def chain_config(system_prompt, *, run_name: str, **metadata) -> dict:
    return {
        "callbacks": [get_langfuse_handler()],
        "run_name": run_name,
        "metadata": {
            "langfuse_prompt": system_prompt,
            **metadata,
        },
    }
