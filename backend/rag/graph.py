from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver


from rag_fusion import retrieval_rag_fusion
from fallback import fallback
from generation import generate_chain
from graders import retrieval_grader, answer_grader, hallucination_grader
from web_search import web_search_tool
from routing import route_question
from core.config import DATA_DIR


class GraphState(TypedDict):
    '''
    Graph State
    '''
    question: str
    generation: str
    documents: List[str]

"""
====================================================
Nodes
"""


def make_retrieve_node(retriever):
    retrieval_chain = retrieval_rag_fusion(retriever)  

    def retrieve(state):
        print("---Retrieving...---")
        question = state["question"]
        documents = retrieval_chain.invoke({"question": question})
        return {"documents": documents, "question": question}

    return retrieve

def llm_fallback(state):
    print("---Fallback...---")
    question = state["question"]
    generation = fallback().invoke({"question": question})
    return {"question": question, "generation": generation}

def generate(state):
    print("---Generating...---")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]
        
    generated_output = generate_chain().invoke({"question": question, "documents": documents})
    return {"question": question, "generation": generated_output, "documents": documents}


def grade_documents(state):
    print("---Checking document relevance...---")
    question = state["question"]
    documents = state["documents"]

    if not isinstance(documents, list):
        documents = [documents]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader().invoke({"question": question, "document": d})
        grade = score.is_relevant
        if grade == "yes":
            print("---GRADE: Document relevant---")
            filtered_docs.append(d)
        else:
            print("---GRADE: Document not relevant---")
            continue
    return {"documents": filtered_docs, "question": question}

def web_search(state):
    print("---Web Searching...---")
    question = state["question"]

    docs = web_search_tool.invoke({"query": question}) 
    web_search_doc = []
    for d in docs["results"]:
        doc = Document(page_content=str(d["content"]),
                        metadata={
                            "source":d["url"]
                        })
        web_search_doc.append(doc)
        
    return { "documents": web_search_doc, "question": question}


"""
=============================================================
Edges
"""

def route_question(state):
    print("---Routing question...---")
    question = state["question"]
    source = route_question().invoke({"question": question})

    if "tool_calls" not in source.additional_kwargs:
        print("---Route question to fallback---")
        return "llm_fallback"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise "Route could no decide source"

    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "web_search":
        print("---Route question to web search---")
        return "web_search"
    elif datasource == "vectorstore":
        print("---Route question to RAG---")
        return "vectorstore"
    else:
        print('f{datasource}: Route undefined, route to web search---')
        return "web_search"



def decide_to_generate(state):
    print("---Decide to generate---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("---No related document, do web search---")
        return "web_search"
    else:

        print("---Document related, generate---")
        return "generate"


def check_hallucination_and_utility(state):
    print("---Check hallucination and utility---")
    question = state["question"]
    document = state["documents"]
    generation = state["generation"]

    check_hallucination = hallucination_grader().invoke({"document": document, "generation": generation})
    is_grounded = check_hallucination.is_grounded

    if is_grounded == "yes":
        print("---Generation is grounded---")
        print("---Check utility---")
        check_utility = answer_grader().invoke({"question": question, "generation": generation})
        has_answered = check_utility.has_answered
        if has_answered == "yes":
            print("---Generation addresses question---")
            return "useful"
        else:
            print("---Generation not useful, web search again---")
            return "not useful"
    else:
        print("---Generation is not grounded in documents, regeneration---")
        return "not supported"

"""
=====================================================================
Build Graph
"""

def build_graph(retriever):
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search)
    retrival_node = make_retrieve_node(retriever)
    workflow.add_node("retrieve", retrival_node)
    workflow.add_node("grade_document", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("llm_fallback", llm_fallback)

    workflow.add_conditional_edges(
        START, 
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
            "llm_fallback": "llm_fallback",
        },
    )

    workflow.add_edge("web_search", "grade_document")
    workflow.add_edge("retrieve", "grade_document")
    workflow.add_edge("llm_fallback", END)

    workflow.add_conditional_edges(
        "grade_document",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "generate",
        check_hallucination_and_utility,
        {
            "useful": END,
            "not useful": "web_search",
            "not supported": "generate",
        },
    )

    checkpointer = SqliteSaver.from_conn_string(str(DATA_DIR/"checkpoints.db"))
    return workflow.compile(checkpointer=checkpointer)



    

