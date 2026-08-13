from rag.vectorstore import VectorStore


def get_retriever():
    return VectorStore("summaries").retriever
