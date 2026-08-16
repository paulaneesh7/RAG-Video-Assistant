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



