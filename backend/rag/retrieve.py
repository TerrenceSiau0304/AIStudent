from vectorstore import VectorStore


def get_retrieve():
    return VectorStore("summaries").retriever
