"""Orchestrate retrieval and grounded generation for user questions."""

from typing import Any, Callable

from src.answer_generator import generate_grounded_answer
from src.rag_output import build_rag_output


def _validate_question_request(question: str, index: dict[str, Any], top_k: int) -> str:
    """Validate a question request and return its normalized text."""
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if not index:
        raise ValueError("Vector index cannot be empty.")

    if top_k < 2 or top_k > 5:
        raise ValueError(
            "top_k must be between 2 and 5."
        )

    return clean_question


def _retrieve_question_chunks(
    question: str,
    index: dict[str, Any],
    client: Any,
    embedding_model: str,
    embedding_function: Callable,
    retrieval_function: Callable,
    top_k: int,
) -> list[dict[str, Any]]:
    """Embed a question and retrieve its most similar chunks."""
    query_embedding = embedding_function(
        text=question,
        client=client,
        model=embedding_model,
    )

    chunks = retrieval_function(
        query_embedding=query_embedding,
        index=index,
        top_k=top_k,
    )

    if len(chunks) < 2:
        raise ValueError(
            "At least two chunks must be retrieved."
        )

    return chunks


def _build_answer_output(
    question: str,
    chunks: list[dict[str, Any]],
    client: Any,
    llm_model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Generate an answer and build its validated public output."""
    answer = generate_grounded_answer(
        question=question,
        chunks=chunks,
        client=client,
        model=llm_model,
        max_output_tokens=max_output_tokens,
    )

    return build_rag_output(
        question,
        answer,
        chunks,
    )


def answer_question(
    question: str,
    index: dict[str, Any],
    client: Any,
    embedding_model: str,
    llm_model: str,
    max_output_tokens: int,
    create_query_embedding: Callable,
    retrieve_chunks: Callable,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run the complete RAG flow and return a validated result."""
    clean_question = _validate_question_request(
        question,
        index,
        top_k,
    )
    chunks = _retrieve_question_chunks(
        clean_question, index, client, embedding_model,
        create_query_embedding, retrieve_chunks, top_k,
    )

    return _build_answer_output(
        clean_question, chunks, client,
        llm_model, max_output_tokens,
    )