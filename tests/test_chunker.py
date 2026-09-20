"""Tests for semantic document chunking."""

import pytest

from src.chunker import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_document,
    count_tokens,
    parse_document_blocks,
    split_sentences,
)
from src.config import DOCUMENT_PATH
from src.document_loader import load_document


def test_count_tokens_returns_positive_value() -> None:
    """Token counting should detect tokens in non-empty text."""
    assert count_tokens("AR HR support documentation.") > 0


def test_split_sentences_preserves_punctuation() -> None:
    """Sentence splitting should preserve sentence-ending punctuation."""
    text = "First sentence. Second sentence? Third sentence!"

    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence?",
        "Third sentence!",
    ]


def test_parse_document_blocks_tracks_section_titles() -> None:
    """Paragraphs should retain the section where they appeared."""
    text = (
        "AR HR GUIDE\n\n"
        "Introduction paragraph.\n\n"
        "## Account Access\n\n"
        "Employees access the platform with their email.\n\n"
        "## Payroll\n\n"
        "Payslips are published every month."
    )

    blocks = parse_document_blocks(text)

    assert blocks == [
        ("Introduction", "AR HR GUIDE"),
        ("Introduction", "Introduction paragraph."),
        (
            "Account Access",
            "Employees access the platform with their email.",
        ),
        (
            "Payroll",
            "Payslips are published every month.",
        ),
    ]


def test_real_document_produces_at_least_20_chunks() -> None:
    """The source document should produce the required chunk count."""
    document = load_document(DOCUMENT_PATH)

    chunks = chunk_document(document)

    assert len(chunks) >= 20


def test_real_document_chunks_respect_token_limits() -> None:
    """Every generated chunk should satisfy the rubric token limits."""
    document = load_document(DOCUMENT_PATH)

    chunks = chunk_document(document)

    assert all(
        MIN_CHUNK_TOKENS <= chunk["token_count"] <= MAX_CHUNK_TOKENS
        for chunk in chunks
    )


def test_real_document_chunk_ids_are_unique() -> None:
    """Every chunk should have a unique deterministic identifier."""
    document = load_document(DOCUMENT_PATH)

    chunks = chunk_document(document)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids[0] == "chunk_001"
    assert chunk_ids[-1] == f"chunk_{len(chunks):03d}"


def test_real_document_chunks_include_section_metadata() -> None:
    """Every chunk should preserve its source section."""
    document = load_document(DOCUMENT_PATH)

    chunks = chunk_document(document)

    assert all(chunk["section"] for chunk in chunks)
    assert all(chunk["text"] for chunk in chunks)


def test_empty_document_is_rejected() -> None:
    """Chunking should reject an empty document."""
    with pytest.raises(ValueError, match="empty"):
        chunk_document("")