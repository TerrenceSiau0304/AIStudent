import uuid
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

class VectorStore:
    def __init__(self, collection_name: str, embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")):
        self.retriever = MultiVectorRetriever(
            vectorstore=Chroma(collection_name=collection_name, embedding_function=embedding_function),
            byte_store=InMemoryByteStore(),
            id_key="doc_id"
        )

    def setup_retriever(self, original_document, summary):
        doc_ids = [str(uuid.uuid4()) for _ in original_document]
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i,s in enumerate(summary)
        ]

        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, original_document)))
        
