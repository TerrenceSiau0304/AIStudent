import pickle
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR/"data"

# def summarise_document(model: str = 'llama3.1', temperature: float = 0):
#     llm = ChatOllama(model, max_retries=0, temperature=temperature)
#     summarise_prompt = ChatPromptTemplate.from_template("Summarise the following document: \n\n{doc}")


#     return (
#         {"doc": lambda x:x.page_content}
#         | summarise_prompt
#         | llm
#         | StrOutputParser()
#     )

def save_summary(file: str, summaries: list):
    path = DATA_DIR/file
    with open(path, "wb") as f:
        pickle.dump(summaries, f)


def load_summary(file: str):
    path = DATA_DIR/file
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results
