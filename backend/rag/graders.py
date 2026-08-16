from typing import Literal
from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_settings

class GradeDocuments(BaseModel):
    """
    Structured output used to represent the relevance of a
    retrieved document to the user's question.

    Attributes:
        is_relevant (Literal['yes', 'no']):
            Indicates whether the retrieved document contains
            information that is relevant to the user's question.
    """
    is_relevant: Literal['yes', 'no'] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def retrieval_grader(model: str="command-r-plus-08-2024", temperature: float=0):
    """
    Create a document relevance grading chain.

    The grader evaluates whether a retrieved document is relevant
    to the user's question based on keyword matches and semantic
    similarity.

    Cohere's language model is used with structured output so that
    the result is restricted to the `GradeDocuments` schema.

    Args:
        model (str):
            Name of the Cohere model used to evaluate document
            relevance.

        temperature (float):
            Controls the randomness of the model's response.
            A value of 0 produces more deterministic grading.

    Returns:
        Runnable:
            A LangChain runnable that accepts a retrieved document
            and user question and returns a `GradeDocuments` object.
    """

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
    """
    Structured output used to determine whether an LLM-generated
    answer is supported by the provided facts or documents.

    Attributes:
        is_grounded (Literal['yes', 'no']):
            Indicates whether the generated answer is grounded
            in the information provided by the retrieved documents.
    """
    is_grounded: Literal['yes', 'no'] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def hallucination_grader(model: str="command-r-plus-08-2024", temperature: float=0):
    """
    Create a hallucination grading chain.

    The grader evaluates whether an LLM-generated answer is
    supported by the provided set of facts or retrieved documents.

    This helps determine whether the generated answer contains
    information that is not supported by the retrieved context.

    Cohere's language model is configured to return structured
    output using the `GradeHallucination` schema.

    Args:
        model (str):
            Name of the Cohere model used to evaluate whether
            the generated answer is grounded in the provided facts.

        temperature (float):
            Controls the randomness of the model's response.
            A value of 0 produces more deterministic grading.

    Returns:
        Runnable:
            A LangChain runnable that accepts a set of facts and
            an LLM-generated answer, then returns a
            `GradeHallucination` object.
    """

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
    """
    Structured output used to determine whether an LLM-generated
    answer adequately addresses the user's question.

    Attributes:
        has_answered (Literal['yes', 'no']):
            Indicates whether the generated answer resolves or
            sufficiently addresses the user's question.
    """
    has_answered: Literal['yes', 'no']= Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def answer_grader(model: str="command-r-plus-08-2024", temperature: float=0):
    """
    Create an answer evaluation chain.

    The grader evaluates whether an LLM-generated answer
    sufficiently addresses the user's original question.

    Unlike the hallucination grader, which checks whether an
    answer is supported by retrieved documents, this grader
    focuses on whether the answer actually resolves the user's
    question.

    Cohere's language model is configured to return structured
    output using the `AnswerGrader` schema.

    Args:
        model (str):
            Name of the Cohere model used to evaluate the
            generated answer.

        temperature (float):
            Controls the randomness of the model's response.
            A value of 0 produces more deterministic grading.

    Returns:
        Runnable:
            A LangChain runnable that accepts a user question
            and an LLM-generated answer, then returns an
            `AnswerGrader` object.
    """
    preamble="""You are a grader assessing whether an answer addresses/ resolves a question. \n
    Answer with 'yes' or 'no' to indicate whether the answer resolves the question."""

    answer_grader_llm = ChatCohere(model=model, temperature=temperature, cohere_api_key=get_settings().cohere_api_key).with_structured_output(AnswerGrader, preamble=preamble)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ]
    )

    return (answer_prompt | answer_grader_llm)



