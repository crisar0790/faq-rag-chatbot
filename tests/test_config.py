"""Tests for the base project configuration."""

import pytest

from pathlib import Path

from src import config


def test_project_paths_are_based_on_repository_root() -> None:
    """Project paths should be derived from the repository root."""
    expected_root = Path(__file__).resolve().parent.parent

    assert config.BASE_DIR == expected_root
    assert config.DATA_DIR == expected_root / "data"
    assert config.DOCUMENT_PATH == expected_root / "data" / "faq_document.txt"
    assert config.INDEX_PATH == expected_root / "data" / "index.json"


def test_embedding_model_has_default_value(monkeypatch) -> None:
    """The embedding model should have a documented default."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    assert config.get_embedding_model() == "text-embedding-3-small"

def test_openai_timeout_has_default_value(monkeypatch) -> None:
    """The API timeout should have a positive default."""
    monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)

    assert config.get_openai_timeout() == 30


def test_invalid_openai_timeout_raises(monkeypatch) -> None:
    """A nonnumeric API timeout should be rejected."""
    monkeypatch.setenv("OPENAI_TIMEOUT", "invalid")

    with pytest.raises(ValueError, match="OPENAI_TIMEOUT"):
        config.get_openai_timeout()


def test_nonpositive_openai_timeout_raises(monkeypatch) -> None:
    """The API timeout must be greater than zero."""
    monkeypatch.setenv("OPENAI_TIMEOUT", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        config.get_openai_timeout()


def test_missing_api_key_raises(monkeypatch) -> None:
    """The OpenAI client requires an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        config.get_openai_client()