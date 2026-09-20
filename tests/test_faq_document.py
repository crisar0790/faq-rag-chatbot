"""Tests for the FAQ source document."""

from src.config import DOCUMENT_PATH


def read_document() -> str:
    """Read the support document using UTF-8 encoding."""
    return DOCUMENT_PATH.read_text(encoding="utf-8")


def test_document_has_at_least_1000_words() -> None:
    """The source document should satisfy the minimum required length."""
    document = read_document()

    assert len(document.split()) >= 1000


def test_document_contains_multiple_support_sections() -> None:
    """The source document should cover multiple support topics."""
    document = read_document()
    sections = [
        line
        for line in document.splitlines()
        if line.startswith("## ")
    ]

    assert len(sections) >= 10