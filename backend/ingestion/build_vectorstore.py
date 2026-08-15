from ingestion.load_docs import build_document, load_document
from ingestion.summarise import load_summary

from rag.vectorstore import VectorStore
def build_vectorstore():
    original_docs = load_document("result_dict.pkl")
    document_type_docs = build_document(original_docs)
    summaries = load_summary("summaries.pkl")
    #Use cohere online embedding to reduce ram usage in render.com
    ai_vectorstore = VectorStore("summaries")
    ai_vectorstore.setup_retriever(original_document=document_type_docs, summary=summaries)
    print("Vectorstore is built")



if "__name__" == "__main__":
    build_vectorstore()





        
