from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def generate_multiple_queries(model: str = "gemini-3.1-flash-lite", temperature: float = 0):
    rag_template =  """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(rag_template)

    return (
        prompt_rag_fusion
        | ChatGoogleGenerativeAI(model=model, temperature=temperature)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )


def reciprocal_rank_fusion(results: list[list], k=60):
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

    return (
        generate_multiple_queries
        | retriever.map()
        | reciprocal_rank_fusion
    )



