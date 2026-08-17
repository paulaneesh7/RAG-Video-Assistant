import os
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate


PROMPT_LABEL = "production"


def get_langfuse():
    return get_client()


def get_langfuse_handler() -> CallbackHandler:
    return CallbackHandler()


def get_text_prompt(name: str, *, label: str = PROMPT_LABEL):
    return get_langfuse().get_prompt(name, label=label, type="text")



def get_chat_prompt_from_pair(system_prompt: str, user_prompt: str, *, label: str = PROMPT_LABEL):
    system_prompt = get_text_prompt(system_prompt, label=label)
    user_prompt = get_text_prompt(user_prompt, label=label)

    return system_prompt, user_prompt



def chain_config(system_prompt, *, run_name: str, **metadata) -> dict:
    return {
        "callbacks": [get_langfuse_handler],
        "run_name": run_name,
        "metadata": {
            "langfuse_prompt": system_prompt,
            **metadata
        }
    }