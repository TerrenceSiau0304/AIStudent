from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings

def fallback(model: str = "gemini-3.1-flash-lite", temperature: float=0):
    """
    Create a fallback language model chain for answering questions
    using the model's general knowledge.

    This chain is used when the primary StudentAI retrieval or
    web-searching process cannot provide a suitable answer.

    The chain takes a question as input, formats it using the
    fallback prompt, sends it to Google's Gemini model, and
    converts the model output into a plain string.

    Args:
        model (str):
            Name of the Google Gemini model used for generating
            the fallback response.

        temperature (float):
            Controls the randomness of the model's response.
            A value of 0 produces more deterministic responses.

    Returns:
        Runnable:
            A LangChain runnable pipeline that accepts a question
            and returns the generated answer as a string.
    """

    fallback_template ="""You are an assistant for question-answering tasks. Answer the question based upon your knowledge.
    Use two sentences and keep the answer concise\n\n

    Question: {question}
    """

    fallback_prompt = ChatPromptTemplate.from_template(fallback_template)

    fallback_llm = ChatGoogleGenerativeAI(temperature=temperature, model=model, google_api_key=get_settings().google_api_key)

    return (
        {"question": itemgetter("question")}
        | fallback_prompt
        | fallback_llm
        | StrOutputParser()
    )
