from typing import Literal
from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_settings

class GradeDocuments(BaseModel):
    is_relevant: Literal['yes', 'no'] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def retrieval_grader(model: str="command-r-plus-08-2024", temperature: float=0):
    preamble="""You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Answer with 'yes' or 'no' to indicate whether the document is relevant to the question."""

    doc_grader_llm = ChatCohere(model=model, temperature=temperature, cohere_api_key=get_settings().cohere_api_key).with_structured_output(GradeDocuments, preamble=preamble)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ]
    )

    return (grade_prompt | doc_grader_llm)

class GradeHallucination(BaseModel):
    is_grounded: Literal['yes', 'no'] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def hallucination_grader(model: str="command-r-plus-08-2024", temperature: float=0):
    preamble="""You are a grader assessing whether an LLM generation is grounded in/ supported by a set of facts. \n
    Answer with 'yes' or 'no' to indicate whether the generation is grounded in/ supported by a set of facts."""

    hallucination_grader_llm = ChatCohere(model=model, temperature=temperature, cohere_api_key=get_settings().cohere_api_key).with_structured_output(GradeHallucination, preamble=preamble)

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Set of facts: \n\n {document} \n\n LLM generation: {generation}")
        ]
    )

    return (hallucination_prompt | hallucination_grader_llm)

class AnswerGrader(BaseModel):
    has_answered: Literal['yes', 'no']= Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def answer_grader(model: str="command-r-plus-08-2024", temperature: float=0):
    preamble="""You are a grader assessing whether an answer addresses/ resolves a question. \n
    Answer with 'yes' or 'no' to indicate whether the answer resolves the question."""

    answer_grader_llm = ChatCohere(model=model, temperature=temperature, cohere_api_key=get_settings().cohere_api_key).with_structured_output(AnswerGrader, preamble=preamble)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ]
    )

    return (answer_prompt | answer_grader_llm)



