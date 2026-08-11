from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings

def generate_chain(model: str="gemini-3.1-flash-lite", temperature: float=0):
    generation_template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved documents to answer the question.
    Cite the given answer based on the metadata of documents.\n\n

    documents: {documents} /n

    Question: {question}
    """

    generation_prompt = ChatPromptTemplate.from_template(generation_template)

    generative_llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=get_settings().google_api_key)

    return (
        {"documents": itemgetter("documents"),
         "question": itemgetter("question")}
         | generation_prompt
         | generative_llm
         | StrOutputParser()
    )


