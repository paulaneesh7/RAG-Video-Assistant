from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from core.observability.langfuse import PROMPT_NAMES, chain_config, get_chat_prompt_from_pair

load_dotenv()


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def split_transcript(transcript: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(transcript)


def summarize_transcript(transcript: str) -> str:
    llm = get_llm()

    map_system_prompt, map_user_prompt = get_chat_prompt_from_pair(
        PROMPT_NAMES["summarize_map_system"],
        PROMPT_NAMES["summarize_map_user"],
    )
    reduce_system_prompt, reduce_user_prompt = get_chat_prompt_from_pair(
        PROMPT_NAMES["summarize_reduce_system"],
        PROMPT_NAMES["summarize_reduce_user"],
    )

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", map_system_prompt.get_langchain_prompt()),
            ("human", map_user_prompt.get_langchain_prompt()),
        ]
    )
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", reduce_system_prompt.get_langchain_prompt()),
            ("human", reduce_user_prompt.get_langchain_prompt()),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summaries = [
        map_chain.invoke(
            {"text": chunk},
            config=chain_config(
                map_system_prompt,
                run_name="map_summarize",
                chunk_index=i,
            ),
        )
        for i, chunk in enumerate(chunks)
    ]

    combined_chain = combined_prompt | llm | StrOutputParser()
    return combined_chain.invoke(
        {"text": "\n\n".join(chunk_summaries)},
        config=chain_config(reduce_system_prompt, run_name="reduce_summarize"),
    )


def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_system_prompt, title_user_prompt = get_chat_prompt_from_pair(
        PROMPT_NAMES["meeting_title_system"],
        PROMPT_NAMES["meeting_title_user"],
    )
    title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", title_system_prompt.get_langchain_prompt()),
            ("human", title_user_prompt.get_langchain_prompt()),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    return title_chain.invoke(
        {"text": transcript[:2000]},
        config=chain_config(title_system_prompt, run_name="meeting_title"),
    )
