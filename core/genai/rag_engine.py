from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langfuse import observe

from core.genai.vector_store import build_vector_store, get_retriever, load_vector_store
from core.observability.langfuse import (
    PROMPT_NAMES,
    chain_config,
    get_chat_prompt_from_pair,
    get_llm_from_prompt,
)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


class TracedRagChain:
    def __init__(self, chain, system_prompt):
        self.chain = chain
        self.langfuse_system_prompt = system_prompt

    def invoke(self, question: str, config=None):
        return self.chain.invoke(question, config=config)


def _rag_prompt():
    system_prompt, user_prompt = get_chat_prompt_from_pair(
        PROMPT_NAMES["rag_system"],
        PROMPT_NAMES["rag_user"],
    )
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.get_langchain_prompt()),
            ("human", user_prompt.get_langchain_prompt()),
        ]
    )
    return system_prompt, template


def assemble_chain(retriever, system_prompt, prompt):
    llm = get_llm_from_prompt(system_prompt, fallback_temperature=0.4)
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return TracedRagChain(rag_chain, system_prompt)


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=5)
    system_prompt, prompt = _rag_prompt()
    return assemble_chain(retriever, system_prompt, prompt)


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    system_prompt, prompt = _rag_prompt()
    return assemble_chain(retriever, system_prompt, prompt)


@observe(name="rag_qa")
def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(
        question,
        config=chain_config(rag_chain.langfuse_system_prompt, run_name="rag_qa"),
    )
    print(f"answer :{answer}")
    return answer
