"""Load and normalize plain-text support documents."""

import re
import unicodedata
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize Unicode, whitespace, and line endings."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")

    lines = [
        " ".join(line.split())
        for line in normalized.split("\n")
    ]
    cleaned = "\n".join(lines)

    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_document(path: Path) -> str:
    """Read and clean a UTF-8 plain-text document."""
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Document must use UTF-8 encoding: {path}"
        ) from error

    document = clean_text(content)

    if not document:
        raise ValueError(f"Document is empty: {path}")

    return document