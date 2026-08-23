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
    "get_llm_from_prompt",
    "chain_config",
]


def env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


client: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global client
    if client is None:
        client = Langfuse(
            public_key=env("LANGFUSE_PUBLIC_KEY"),
            secret_key=env("LANGFUSE_SECRET_KEY"),
            host=env("LANGFUSE_HOST") or env("LANGFUSE_BASE_URL"),
        )
    return client


def get_langfuse_handler() -> CallbackHandler:
    return CallbackHandler(public_key=env("LANGFUSE_PUBLIC_KEY"))


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


def get_llm_from_prompt(prompt, *, fallback_model: str = "gpt-4o-mini", fallback_temperature: float = 0.3):
    """Build ChatOpenAI from Langfuse prompt.config (model, temperature, max_tokens, seed, top_p)."""
    from langchain_openai import ChatOpenAI

    config = getattr(prompt, "config", None) or {}
    if not isinstance(config, dict):
        config = {}

    kwargs = {
        "model": config.get("model", fallback_model),
        "temperature": float(config.get("temperature", fallback_temperature)),
    }
    if config.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(config["max_tokens"])
    if config.get("top_p") is not None:
        kwargs["top_p"] = float(config["top_p"])
    if config.get("seed") is not None:
        kwargs["seed"] = int(config["seed"])
    return ChatOpenAI(**kwargs)


def chain_config(system_prompt, *, run_name: str, **metadata) -> dict:
    return {
        "callbacks": [get_langfuse_handler()],
        "run_name": run_name,
        "metadata": {
            "langfuse_prompt": system_prompt,
            **metadata,
        },
    }
