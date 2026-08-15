import uuid
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
# from langchain_classic.storage import LocalFileStore
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# from core.config import DATA_DIR

#pip install langchain-huggingface removed from requirement.txt for space, (HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))



class VectorStore:
    def __init__(self, collection_name: str, embedding_function=None):
        self.retriever = MultiVectorRetriever(
            vectorstore=Chroma(collection_name=collection_name,
                                embedding_function=embedding_function,
                                # persist_directory=str(DATA_DIR/"chroma_persist")
                                persist_directory=str(Path("/tmp")/"chroma_persist")),
            # byte_store=LocalFileStore(str(DATA_DIR/"docstore")),
            byte_store= InMemoryByteStore(),
            id_key="doc_id"
        )

    def is_empty(self):
        return self.retriever.vectorstore._collection.count() == 0

    def setup_retriever(self, original_document, summary):
        doc_ids = [str(uuid.uuid4()) for _ in original_document]
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i,s in enumerate(summary)
        ]

        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, original_document)))