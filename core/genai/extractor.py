# ActionableItems, Decision, Questions Extractor


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os




def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)




# creating a custom function for chain
def build_chain(system_prompt: str):
    llm = get_llm()

    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{text}")])
        | llm
        | StrOutputParser()
    )

    return chain




def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """
            You're an expert meeting analyst. From the meeting transcript,
            extracy all action items. For each provide:
            - Task description
            - Owner (who is responsible)
            - Deadline (if mentioned, else write 'Not specified')
            Format as a numbered list. If none found say 'No action items found.
        """
    )


    return chain.invoke(transcript)



def extract_key_decision(transcript: str) -> str:
    chain = build_chain(
        """
        
            You're an expert meeting analyst. From the meeting transcript,
            extract all key decisions made. Format as a numbered list.
            If none found say 'No key decisions found.
        """
    )

    return chain.invoke(transcript)




def extract_questions(transcript: str) -> str:

    chain = build_chain(
        """
        
            From the meeting transcript, extract all unresolved questions.
            or topics needing follow-up. Format as a numbered list.
            If none found say 'No open questions found.
        """
    )

    return chain.invoke(transcript)