"""Retrival document chunks using cosine similarity."""

from typing import Any

import numpy as np

MIN_RESULTS = 2
MAX_RESULTS = 5

def cosine_similarity(first_vector: list[float], second_vector: list[float]) ->float:
    """Calculate cosine similarity between two vectors."""
    if len(first_vector) != len(second_vector):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    first = np.asarray(first_vector, dtype=float)
    second = np.asarray(second_vector, dtype=float)

    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)

    if first_norm == 0 or second_norm == 0:
        raise ValueError(
            "Cosine similarity cannot use a zero vector."
        )

    return float(
        np.dot(first, second)
        / (first_norm * second_norm)
    )

def validate_search_input(index: dict[str, Any], query_embedding: list[float], top_k: int) -> None:
    """Validate the index, query vector, and result count."""
    if not MIN_RESULTS <= top_k <= MAX_RESULTS:
        raise ValueError(
            f"TOP_K must be between {MIN_RESULTS} and {MAX_RESULTS}."
        )

    expected_dimension = index.get("embedding_dimension")

    if len(query_embedding) != expected_dimension:
        raise ValueError(
            "Query embedding dimension does not match the index."
        )

    chunks = index.get("chunks", [])

    if len(chunks) < top_k:
        raise ValueError(
            "The index does not contain enough chunks "
            "for the requested TOP_K"
        )

def build_search_result(chunk: dict[str, Any], similarity: float) ->dict[str, Any]:
    """Build a result without exposing stored embedding."""
    return {
        "chunk_id": chunk["chunk_id"],
        "section": chunk["section"],
        "text": chunk["text"],
        "token_count": chunk["token_count"],
        "similarity_score": similarity,
    }

def search_similar_chunks(index: dict[str, Any], query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Return the top-k chunks ordered by cosine similarity."""
    validate_search_input(index=index, query_embedding=query_embedding, top_k=top_k)

    results = [
        build_search_result(
            chunk=chunk,
            similarity=cosine_similarity(
                query_embedding,
                chunk["embedding"],
            ),
        )
        for chunk in index["chunks"]
    ]

    return sorted(
        results,
        key=lambda result: result["similarity_score"],
        reverse=True,
    )[:top_k]