from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core.config import get_settings


class web_search(BaseModel):
    """
    Structured tool definition for routing a question to web search.

    Attributes:
        query (str):
            Search query that will be used to find information
            on the internet.
    """
    query: str = Field(description="The query to use when searching the internet.")

class vectorstore(BaseModel):
    """
    Structured tool definition for routing a question to the
    vector store.

    Attributes:
        query (str):
            Search query that will be used to retrieve information
            from the vector store.
    """
    query: str = Field(description="The query to use when search the vectorstore.")


def route_question_chain(model:str="command-r-plus-08-2024", temperature:float = 0):
    """
    Create a question-routing chain that determines whether a
    user question should be answered using the vector store or
    an external web search.

    The router uses Cohere's tool-calling capability to select
    between the `web_search` and `vectorstore` tools based on
    the subject of the user's question. If tool-calling is false, 
    the question will be route to fallback.

    The vector store contains educational materials covering
    artificial intelligence and object-oriented programming,
    including topics such as intelligent agents, search
    algorithms, probability reasoning, propositional logic,
    Markov Decision Processes, reinforcement learning,
    machine learning, neural networks, ethics, and
    object-oriented programming theory.

    Args:
        model (str):
            Name of the Cohere model used for question routing.

        temperature (float):
            Controls the randomness of the routing model.
            A value of 0 produces more deterministic routing.

    Returns:
        Runnable:
            A LangChain runnable that accepts a user question
            and returns a tool call indicating the selected
            information source.
    """
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

