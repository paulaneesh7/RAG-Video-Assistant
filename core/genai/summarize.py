from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

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


    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a helpful assistant that summarizes transcriptions of a video concisely and accurately."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]


    # Combine the summaries into a single string
    combined_summaries = "\n\n".join(chunk_summaries)


    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You're an expert meeting summarizer. 
            Combine these partial summaries into one final professional meeting summary in bullet points."""),
            ("human", "{text}"),
        ]
    )


    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined_summaries)





def generate_title(transcript: str) -> str:
    llm = get_llm()


    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", """
                Based on the meeting transcript, generate a short professional meeting title
                (max 8 words). Only return the title, no other text.
            """),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


    return title_chain.invoke(transcript[:2000])