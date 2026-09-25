"""Tests for the required RAG output contract."""

import pytest

from src.rag_output import build_rag_output, validate_rag_output


def sample_chunks() -> list[dict]:
    """Return retrieved chunks used by output tests."""
    return [
        {
            "chunk_id": "chunk_001",
            "section": "Account Access",
            "text": "Employees can reset their password.",
            "token_count": 52,
            "similarity_score": 0.91,
        },
        {
            "chunk_id": "chunk_002",
            "section": "Account Access",
            "text": "Password reset links expire after 30 minutes.",
            "token_count": 58,
            "similarity_score": 0.85,
        },
    ]


def test_build_rag_output_contains_exact_required_keys() -> None:
    """The public result should contain exactly three keys."""
    output = build_rag_output(
        question="How can I reset my password?",
        answer="Use the password reset link on the login page.",
        chunks=sample_chunks(),
    )

    assert set(output) == {
        "user_question",
        "system_answer",
        "chunks_related",
    }


def test_build_rag_output_preserves_question_and_answer() -> None:
    """The result should contain the original question and answer."""
    output = build_rag_output(
        question="How can I reset my password?",
        answer="Use the password reset link on the login page.",
        chunks=sample_chunks(),
    )

    assert output["user_question"] == (
        "How can I reset my password?"
    )
    assert output["system_answer"] == (
        "Use the password reset link on the login page."
    )


def test_chunks_related_contains_public_chunk_fields() -> None:
    """Related chunks should expose their source information."""
    output = build_rag_output(
        question="How can I reset my password?",
        answer="Use the password reset link on the login page.",
        chunks=sample_chunks(),
    )

    assert output["chunks_related"][0] == {
        "chunk_id": "chunk_001",
        "section": "Account Access",
        "text": "Employees can reset their password.",
    }


def test_chunks_related_does_not_expose_internal_fields() -> None:
    """Internal retrieval metadata should not enter the public result."""
    output = build_rag_output(
        question="How can I reset my password?",
        answer="Use the password reset link on the login page.",
        chunks=sample_chunks(),
    )

    assert all(
        "embedding" not in chunk
        and "similarity_score" not in chunk
        and "token_count" not in chunk
        for chunk in output["chunks_related"]
    )


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("", "Valid answer."),
        ("   ", "Valid answer."),
        ("Valid question?", ""),
        ("Valid question?", "   "),
    ],
)
def test_build_rag_output_rejects_blank_text(
    question: str,
    answer: str,
) -> None:
    """Questions and answers should contain useful text."""
    with pytest.raises(ValueError, match="must not be blank"):
        build_rag_output(
            question=question,
            answer=answer,
            chunks=sample_chunks(),
        )


@pytest.mark.parametrize("chunk_count", [0, 1, 6])
def test_build_rag_output_requires_two_to_five_chunks(
    chunk_count: int,
) -> None:
    """The result should contain between two and five chunks."""
    chunks = [
        {
            "chunk_id": f"chunk_{index:03d}",
            "section": "Test Section",
            "text": f"Test source text {index}.",
        }
        for index in range(1, chunk_count + 1)
    ]

    with pytest.raises(ValueError, match="between 2 and 5"):
        build_rag_output(
            question="Valid question?",
            answer="Valid answer.",
            chunks=chunks,
        )


def test_validate_rag_output_rejects_extra_top_level_key() -> None:
    """No additional top-level fields should be accepted."""
    output = {
        "user_question": "Valid question?",
        "system_answer": "Valid answer.",
        "chunks_related": [
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "First source.",
            },
            {
                "chunk_id": "chunk_002",
                "section": "Account Access",
                "text": "Second source.",
            },
        ],
        "similarity_score": 0.9,
    }

    with pytest.raises(ValueError, match="exactly"):
        validate_rag_output(output)


def test_validate_rag_output_rejects_duplicate_chunks() -> None:
    """The same source chunk should not appear more than once."""
    output = {
        "user_question": "Valid question?",
        "system_answer": "Valid answer.",
        "chunks_related": [
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "First source.",
            },
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "First source.",
            },
        ],
    }

    with pytest.raises(ValueError, match="must be unique"):
        validate_rag_output(output)