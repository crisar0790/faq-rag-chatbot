import json
from types import SimpleNamespace

import pytest

from src.rag_service import answer_question


class FakeResponsesAPI:
    def create(self, **kwargs):
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(
                {
                    "system_answer": (
                        "Employees can reset their password "
                        "from the AR HR login page. The reset "
                        "link expires after 30 minutes."
                    )
                }
            ),
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponsesAPI()


@pytest.fixture
def vector_index():
    return [
        {
            "chunk_id": "chunk_001",
            "section": "Account Access",
            "text": (
                "Employees can request a password reset "
                "from the login page."
            ),
            "embedding": [0.1, 0.2, 0.3],
            "token_count": 12,
        },
        {
            "chunk_id": "chunk_002",
            "section": "Account Access",
            "text": (
                "Password reset links expire after "
                "30 minutes."
            ),
            "embedding": [0.2, 0.3, 0.4],
            "token_count": 9,
        },
        {
            "chunk_id": "chunk_003",
            "section": "Security",
            "text": (
                "Employees should never share their "
                "password."
            ),
            "embedding": [0.3, 0.4, 0.5],
            "token_count": 8,
        },
    ]


def fake_create_query_embedding(
    text,
    client,
    model,
):
    assert text == "How can I reset my password?"
    assert model == "text-embedding-3-small"

    return [0.1, 0.2, 0.3]


def fake_retrieve_chunks(
    query_embedding,
    index,
    top_k,
):
    assert query_embedding == [0.1, 0.2, 0.3]
    assert top_k == 3

    return index[:top_k]


def test_answer_question_returns_valid_rag_output(
    vector_index,
):
    result = answer_question(
        question="How can I reset my password?",
        index=vector_index,
        client=FakeClient(),
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        max_output_tokens=400,
        create_query_embedding=(
            fake_create_query_embedding
        ),
        retrieve_chunks=fake_retrieve_chunks,
        top_k=3,
    )

    assert set(result) == {
        "user_question",
        "system_answer",
        "chunks_related",
    }

    assert result["user_question"] == (
        "How can I reset my password?"
    )
    assert "30 minutes" in result["system_answer"]
    assert len(result["chunks_related"]) == 3

    first_chunk = result["chunks_related"][0]

    assert set(first_chunk) == {
        "chunk_id",
        "section",
        "text",
    }

    assert "embedding" not in first_chunk
    assert "token_count" not in first_chunk


def test_answer_question_rejects_empty_question(
    vector_index,
):
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        answer_question(
            question="   ",
            index=vector_index,
            client=FakeClient(),
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            max_output_tokens=400,
            create_query_embedding=(
                fake_create_query_embedding
            ),
            retrieve_chunks=fake_retrieve_chunks,
        )


def test_answer_question_rejects_empty_index():
    with pytest.raises(
        ValueError,
        match="Vector index cannot be empty",
    ):
        answer_question(
            question="How can I reset my password?",
            index=[],
            client=FakeClient(),
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            max_output_tokens=400,
            create_query_embedding=(
                fake_create_query_embedding
            ),
            retrieve_chunks=fake_retrieve_chunks,
        )


@pytest.mark.parametrize("top_k", [0, 1, 6])
def test_answer_question_rejects_invalid_top_k(
    vector_index,
    top_k,
):
    with pytest.raises(
        ValueError,
        match="top_k must be between 2 and 5",
    ):
        answer_question(
            question="How can I reset my password?",
            index=vector_index,
            client=FakeClient(),
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            max_output_tokens=400,
            create_query_embedding=(
                fake_create_query_embedding
            ),
            retrieve_chunks=fake_retrieve_chunks,
            top_k=top_k,
        )