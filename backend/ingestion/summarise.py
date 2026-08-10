import pickle

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def summarise_document(model: str = 'llama3.1', temperature: float = 0):
    llm = ChatOllama(model, max_retries=0, temperature=temperature)
    summarise_prompt = ChatPromptTemplate.from_template("Summarise the following document: \n\n{doc}")


    return (
        {"doc": lambda x:x.page_content}
        | summarise_prompt
        | llm
        | StrOutputParser()
    )

def save_summary(file: str, summaries: list):
    with open(file, "wb") as f:
        pickle.dump(summaries, f)


def load_summary(file: str):
    with open(file, "rb") as f:
        results = pickle.load(f)
    return results
