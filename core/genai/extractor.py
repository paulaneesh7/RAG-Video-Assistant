from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.observability.langfuse import (
    PROMPT_NAMES,
    chain_config,
    get_llm_from_prompt,
    get_text_prompt,
)


def build_chain(system_prompt_name: str, *, run_name: str):
    system_prompt = get_text_prompt(system_prompt_name)
    llm = get_llm_from_prompt(system_prompt, fallback_temperature=0.2)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.get_langchain_prompt()),
            ("human", "{text}"),
        ]
    )
    chain = template | llm | StrOutputParser()
    return chain, system_prompt, run_name


def _invoke_extract(system_prompt_name: str, transcript: str, *, run_name: str) -> str:
    chain, system_prompt, name = build_chain(system_prompt_name, run_name=run_name)
    return chain.invoke(
        {"text": transcript},
        config=chain_config(system_prompt, run_name=name),
    )


def extract_action_items(transcript: str) -> str:
    return _invoke_extract(
        PROMPT_NAMES["action_items_system"],
        transcript,
        run_name="extract_action_items",
    )


def extract_key_decision(transcript: str) -> str:
    return _invoke_extract(
        PROMPT_NAMES["key_decision_system"],
        transcript,
        run_name="extract_key_decisions",
    )


def extract_questions(transcript: str) -> str:
    return _invoke_extract(
        PROMPT_NAMES["questions_system"],
        transcript,
        run_name="extract_questions",
    )
