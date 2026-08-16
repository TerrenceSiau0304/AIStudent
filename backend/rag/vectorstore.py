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
    """
    Manage the vector store and document retriever used by the
    StudentAI retrieval system.

    The vector store uses Chroma to store embeddings of document
    summaries and a MultiVectorRetriever to connect those summaries
    back to their corresponding original documents.

    Attributes:
        retriever (MultiVectorRetriever):
            Retriever used to search the vector store and retrieve
            the original documents associated with matching summaries.
    """
    def __init__(self, collection_name: str, embedding_function=None):
        """
        Initialize the vector store and configure its retriever.

        Args:
            collection_name (str):
                Name of the Chroma collection used to store the
                document embeddings.

            embedding_function:
                Embedding function used to convert document
                summaries into vector representations.
        """
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
        """
        Check whether the Chroma vector store contains any
        documents.

        Returns:
            bool:
                True if the vector store contains no documents;
                otherwise False.
        """
        return self.retriever.vectorstore._collection.count() == 0

    def setup_retriever(self, original_document, summary):
        """
        Populate the vector store with document summaries and
        associate them with their original documents.

        Each original document is assigned a unique identifier.
        The summaries are stored in Chroma for similarity search,
        while the original documents are stored in the document
        store.

        Args:
            original_document:
                List of original documents that will be returned
                when their corresponding summaries are retrieved.

            summary:
                List of summaries used to create vector embeddings
                and perform similarity-based retrieval.
        """
        doc_ids = [str(uuid.uuid4()) for _ in original_document]
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i,s in enumerate(summary)
        ]

        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, original_document)))