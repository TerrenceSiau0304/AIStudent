from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings

def fallback(model: str = "gemini-3.1-flash-lite", temperature: float=0):
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
