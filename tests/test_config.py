"""Tests for the base project configuration."""

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