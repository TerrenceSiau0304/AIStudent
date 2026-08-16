from rag.vectorstore import VectorStore


def get_retriever():
    """
    Retrieve the configured retriever from the vector store.

    This function creates a `VectorStore` instance using the
    "summaries" vector store and returns its retriever for use
    in the document retrieval pipeline.

    Returns:
        Retriever:
            The retriever associated with the "summaries"
            vector store.
    """
    return VectorStore("summaries").retriever
