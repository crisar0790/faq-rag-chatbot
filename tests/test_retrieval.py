"""Tests for vector similarity search."""

import pytest

from src.retrieval import cosine_similarity, search_similar_chunks


def sample_index() -> dict:
    """Return a small vector index for retrieval tests."""
    return {
        "index_version": 1,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 3,
        "chunks": [
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "Employees can reset their password.",
                "token_count": 52,
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "chunk_id": "chunk_002",
                "section": "Payroll",
                "text": "Payslips are published every month.",
                "token_count": 54,
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "chunk_id": "chunk_003",
                "section": "Security",
                "text": "Two-factor authentication protects accounts.",
                "token_count": 58,
                "embedding": [0.8, 0.2, 0.0],
            },
        ],
    }


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    """Identical vectors should have maximum similarity."""
    similarity = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_orthogonal_vectors() -> None:
    """Orthogonal vectors should have zero similarity."""
    similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


def test_cosine_similarity_rejects_different_dimensions() -> None:
    """Both vectors must have the same dimension."""
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_cosine_similarity_rejects_zero_vector() -> None:
    """Cosine similarity is undefined for a zero vector."""
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 2.0],
        )


def test_search_returns_chunks_ordered_by_similarity() -> None:
    """Search results should be ordered from most to least similar."""
    results = search_similar_chunks(
        index=sample_index(),
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
    )

    assert [
        result["chunk_id"]
        for result in results
    ] == [
        "chunk_001",
        "chunk_003",
        "chunk_002",
    ]


def test_search_returns_requested_number_of_chunks() -> None:
    """Search should return the configured number of chunks."""
    results = search_similar_chunks(
        index=sample_index(),
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_search_result_does_not_expose_embedding() -> None:
    """Retrieved chunks should not include their embedding vectors."""
    results = search_similar_chunks(
        index=sample_index(),
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert all(
        "embedding" not in result
        for result in results
    )


@pytest.mark.parametrize("top_k", [1, 6])
def test_search_rejects_top_k_outside_required_range(
    top_k: int,
) -> None:
    """TOP_K must remain between two and five."""
    with pytest.raises(ValueError, match="between 2 and 5"):
        search_similar_chunks(
            index=sample_index(),
            query_embedding=[1.0, 0.0, 0.0],
            top_k=top_k,
        )


def test_search_rejects_wrong_query_dimension() -> None:
    """The query and index embeddings must share a dimension."""
    with pytest.raises(ValueError, match="dimension"):
        search_similar_chunks(
            index=sample_index(),
            query_embedding=[1.0, 0.0],
            top_k=2,
        )