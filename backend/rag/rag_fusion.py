from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_settings


def generate_multiple_queries(model: str = "gemini-3.1-flash-lite", temperature: float = 0):
    """
    Create a query generation chain for RAG Fusion.

    The chain uses a language model to generate multiple search
    queries from a single user question. These alternative queries
    can be used to retrieve information from different perspectives,
    improving the chances of finding relevant documents.

    The generated output is expected to contain four separate
    queries, with each query placed on a new line.

    Args:
        model (str):
            Name of the Google Gemini model used to generate the
            alternative search queries.

        temperature (float):
            Controls the randomness of the generated queries.
            A value of 0 produces more deterministic results.

    Returns:
        Runnable:
            A LangChain runnable pipeline that accepts a question
            and returns a list of generated search queries.
    """

    rag_template =  """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(rag_template)

    return (
        prompt_rag_fusion
        | ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=get_settings().google_api_key)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )


def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Combine and rank documents retrieved from multiple search queries
    using the Reciprocal Rank Fusion (RRF) algorithm.

    RRF assigns a score to each document based on its ranking position
    in the results returned by each query. Documents that appear highly
    ranked across multiple query results receive higher overall scores.

    Args:
        results (list[list]):
            A list containing the retrieval results for each generated
            search query. Each inner list contains documents ranked
            according to their relevance to that query.

        k (int):
            Constant used to reduce the influence of the document's
            exact ranking position. The default value is 60.

    Returns:
        list:
            A list of tuples containing each document and its
            calculated RRF score, sorted from highest to lowest score.
    """
    fused_scores={}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1/ (rank+1+k)

    ranked_results = [
        (loads(doc),score)
        for doc, score in sorted(fused_scores.items(), key=lambda x:x[1], reverse=True)
    ]

    return ranked_results

def retrieval_rag_fusion(retriever):
    """
    Create a RAG Fusion retrieval pipeline.

    The pipeline generates multiple search queries from the
    user's original question, performs retrieval for each query,
    and combines the retrieved results using Reciprocal Rank
    Fusion.

    Args:
        retriever:
            Retriever used to search the vector store for documents
            relevant to each generated query.

    Returns:
        Runnable:
            A LangChain runnable pipeline that generates multiple
            queries, retrieves documents for each query, and returns
            the documents ranked using Reciprocal Rank Fusion.
    """

    return (
        generate_multiple_queries()
        | retriever.map()
        | reciprocal_rank_fusion
    )



