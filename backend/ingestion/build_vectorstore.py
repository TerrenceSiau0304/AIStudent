from ingestion.load_docs import build_document, load_document
from ingestion.summarise import load_summary
from langchain_cohere import CohereEmbeddings

from rag.vectorstore import VectorStore
from core.config import get_settings
def build_vectorstore():
    """
    Build and configure the vector store used by the StudentAI
    retrieval system.

    The function loads the processed documents and their summaries,
    creates a vector store using Cohere's online embedding model,
    and configures the retriever with both the original documents
    and their summaries.

    Cohere's online embedding service is used instead of loading
    a local embedding model. This helps reduce RAM consumption
    when deploying the application on resource-limited platforms
    such as Render.com.

    Returns:
        VectorStore:
            A fully configured vector store containing the document
            embeddings and retriever configuration.
    """
    original_docs = load_document("result_dict.pkl")
    document_type_docs = build_document(original_docs)
    summaries = load_summary("summaries.pkl")
    #Use cohere online embedding to reduce ram usage in render.com
    ai_vectorstore = VectorStore("summaries",
                                  CohereEmbeddings(model="embed-english-v3.0",
                                                    cohere_api_key=get_settings().cohere_api_key))
    ai_vectorstore.setup_retriever(original_document=document_type_docs, summary=summaries)
    print("Vectorstore is built")
    return ai_vectorstore



if "__name__" == "__main__":
    build_vectorstore()





        
