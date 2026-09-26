from unittest.mock import Mock

from src.query import (
    create_query_embedding,
    retrieve_chunks,
)


def test_create_query_embedding(monkeypatch):
    client = Mock()

    def fake_generate_embeddings(
        texts,
        client,
        model,
    ):
        assert texts == ["How can I reset my password?"]
        assert model == "text-embedding-3-small"

        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        "src.query.generate_embeddings",
        fake_generate_embeddings,
    )

    result = create_query_embedding(
        text="How can I reset my password?",
        client=client,
        model="text-embedding-3-small",
    )

    assert result == [0.1, 0.2, 0.3]


def test_retrieve_chunks(monkeypatch):
    index = {
        "chunks": [
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "Password reset information.",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    }

    expected_chunks = [
        {
            "chunk_id": "chunk_001",
            "section": "Account Access",
            "text": "Password reset information.",
            "similarity": 0.98,
        }
    ]

    def fake_search_similar_chunks(
        index,
        query_embedding,
        top_k,
    ):
        assert query_embedding == [0.1, 0.2, 0.3]
        assert top_k == 3

        return expected_chunks

    monkeypatch.setattr(
        "src.query.search_similar_chunks",
        fake_search_similar_chunks,
    )

    result = retrieve_chunks(
        query_embedding=[0.1, 0.2, 0.3],
        index=index,
        top_k=3,
    )

    assert result == expected_chunks