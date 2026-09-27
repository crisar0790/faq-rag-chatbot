from unittest.mock import Mock

from src.query import (
    create_query_embedding,
    retrieve_chunks,
    create_query_runner,
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

def test_query_runner_reuses_loaded_resources(
    monkeypatch,
):
    load_index_mock = Mock(
        return_value={
            "embedding_model": (
                "text-embedding-3-small"
            ),
            "chunks": [],
        }
    )
    client = Mock()
    answer_question_mock = Mock(
        side_effect=[
            {"user_question": "First"},
            {"user_question": "Second"},
        ]
    )

    monkeypatch.setattr(
        "src.query.load_index",
        load_index_mock,
    )
    monkeypatch.setattr(
        "src.query.validate_index_model",
        Mock(),
    )
    monkeypatch.setattr(
        "src.query.get_openai_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(
        "src.query.get_embedding_model",
        Mock(
            return_value="text-embedding-3-small"
        ),
    )
    monkeypatch.setattr(
        "src.query.get_llm_model",
        Mock(return_value="gpt-4o-mini"),
    )
    monkeypatch.setattr(
        "src.query.get_max_output_tokens",
        Mock(return_value=400),
    )
    monkeypatch.setattr(
        "src.query.answer_question",
        answer_question_mock,
    )

    runner = create_query_runner()

    first_result = runner("First")
    second_result = runner("Second")

    assert first_result["user_question"] == "First"
    assert second_result["user_question"] == "Second"
    load_index_mock.assert_called_once()
    assert answer_question_mock.call_count == 2