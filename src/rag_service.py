"""Orchestrate retrieval and grounded generation for user questions."""

from typing import Any, Callable

from src.answer_generator import generate_grounded_answer
from src.rag_output import build_rag_output

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
    """Run the complete RAG flow and return a validated public result."""
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if not index:
        raise ValueError("Vector index cannot be empty.")

    if top_k < 2 or top_k > 5:
        raise ValueError("top_k must be between 2 and 5.")

    query_embedding = create_query_embedding(
        text=clean_question,
        client=client,
        model=embedding_model,
    )

    retrieved_chunks = retrieve_chunks(
        query_embedding=query_embedding,
        index=index,
        top_k=top_k,
    )

    if len(retrieved_chunks) < 2:
        raise ValueError(
            "At least two chunks must be retrieved."
        )

    system_answer = generate_grounded_answer(
        question=clean_question,
        chunks=retrieved_chunks,
        client=client,
        model=llm_model,
        max_output_tokens=max_output_tokens,
    )

    return build_rag_output(
        clean_question,
        system_answer,
        retrieved_chunks,
    )