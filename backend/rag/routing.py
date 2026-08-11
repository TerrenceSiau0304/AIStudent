from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core.config import get_settings


class web_search(BaseModel):
    query: str = Field(description="The query to use when searching the internet.")

class vectorstore(BaseModel):
    query: str = Field(description="The query to use when search the vectorstore.")


def route_question(model:str="command-r", temperature:float = 0):
    preamble = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to basic knowledge about artificial intelligence 
    including topics like agent, searching algorithm, probability reasoning,
    propositional logic, Markov Decision Processes, reinforce learning, machine learning, neural network
    and ethics and theory about object-oriented programming.
    Use the vectorstore for a questions on these topics. Otherwise, use web-search."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
        ]
    )

    route_llm = ChatCohere(model=model, temperature=temperature, cohere_api_key=get_settings().cohere_api_key)
    structured_llm_router = route_llm.bind_tools(
        tools=[web_search, vectorstore], preamble=preamble
    )

    return (
        route_prompt | structured_llm_router
    )

