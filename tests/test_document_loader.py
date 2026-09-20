"""Tests for document loading and normalization."""

from pathlib import Path

import pytest

from src.config import DOCUMENT_PATH
from src.document_loader import clean_text, load_document


def test_load_document_reads_the_real_source_file() -> None:
    """The loader should read the project support document."""
    document = load_document(DOCUMENT_PATH)

    assert document.startswith("AR HR SUPPORT AND OPERATIONS GUIDE")
    assert len(document.split()) >= 1000


def test_load_document_supports_utf8_bom(tmp_path: Path) -> None:
    """The loader should support UTF-8 files with a byte order mark."""
    document_path = tmp_path / "document.txt"
    document_path.write_text(
        "Support documentation with accented text: configuración.",
        encoding="utf-8-sig",
    )

    document = load_document(document_path)

    assert document == (
        "Support documentation with accented text: configuración."
    )


def test_load_document_raises_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """A missing document should produce a clear error."""
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Document not found"):
        load_document(missing_path)


def test_load_document_raises_when_file_is_empty(
    tmp_path: Path,
) -> None:
    """A document without useful content should be rejected."""
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("  \n\n  ", encoding="utf-8")

    with pytest.raises(ValueError, match="Document is empty"):
        load_document(empty_path)


def test_load_document_rejects_non_utf8_content(
    tmp_path: Path,
) -> None:
    """The source document should use UTF-8 encoding."""
    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_bytes(b"Invalid UTF-8: \xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        load_document(invalid_path)


def test_clean_text_normalizes_line_endings() -> None:
    """Windows and old Mac line endings should become Unix line endings."""
    text = "First paragraph.\r\n\r\nSecond paragraph.\rThird line."

    cleaned = clean_text(text)

    assert cleaned == (
        "First paragraph.\n\nSecond paragraph.\nThird line."
    )


def test_clean_text_collapses_repeated_spaces() -> None:
    """Repeated horizontal whitespace should become a single space."""
    text = "Ar   HR\tprovides    support."

    assert clean_text(text) == "Ar HR provides support."


def test_clean_text_limits_consecutive_blank_lines() -> None:
    """Multiple blank lines should become one paragraph separator."""
    text = "First paragraph.\n\n\n\nSecond paragraph."

    assert clean_text(text) == (
        "First paragraph.\n\nSecond paragraph."
    )