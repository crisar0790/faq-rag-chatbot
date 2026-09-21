"""Tests for vector index persistence."""

import json
from pathlib import Path

import pytest

from src.vector_store import load_index, save_index


def sample_chunks() -> list[dict]:
    """Return chunks used by vector index tests."""
    return [
        {
            "chunk_id": "chunk_001",
            "section": "Account Access",
            "text": "Employees can reset their password.",
            "token_count": 52,
        },
        {
            "chunk_id": "chunk_002",
            "section": "Payroll",
            "text": "Payslips are published every month.",
            "token_count": 54,
        },
    ]


def sample_embeddings() -> list[list[float]]:
    """Return embeddings used by vector index tests."""
    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_save_index_creates_json_file(tmp_path: Path) -> None:
    """The vector index should be stored as JSON."""
    index_path = tmp_path / "index.json"

    save_index(
        chunks=sample_chunks(),
        embeddings=sample_embeddings(),
        model="text-embedding-3-small",
        path=index_path,
    )

    assert index_path.is_file()


def test_save_index_includes_model_and_dimension(
    tmp_path: Path,
) -> None:
    """The index should describe its embedding configuration."""
    index_path = tmp_path / "index.json"

    save_index(
        chunks=sample_chunks(),
        embeddings=sample_embeddings(),
        model="text-embedding-3-small",
        path=index_path,
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["index_version"] == 1
    assert index["embedding_model"] == "text-embedding-3-small"
    assert index["embedding_dimension"] == 3


def test_save_index_attaches_embedding_to_each_chunk(
    tmp_path: Path,
) -> None:
    """Each stored chunk should include its own embedding."""
    index_path = tmp_path / "index.json"

    save_index(
        chunks=sample_chunks(),
        embeddings=sample_embeddings(),
        model="text-embedding-3-small",
        path=index_path,
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["chunks"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert index["chunks"][1]["embedding"] == [0.4, 0.5, 0.6]


def test_save_index_rejects_different_counts(
    tmp_path: Path,
) -> None:
    """Every chunk must have exactly one embedding."""
    with pytest.raises(ValueError, match="number of embeddings"):
        save_index(
            chunks=sample_chunks(),
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-small",
            path=tmp_path / "index.json",
        )


def test_save_index_rejects_inconsistent_dimensions(
    tmp_path: Path,
) -> None:
    """Stored embeddings should share one dimension."""
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5],
    ]

    with pytest.raises(ValueError, match="same dimension"):
        save_index(
            chunks=sample_chunks(),
            embeddings=embeddings,
            model="text-embedding-3-small",
            path=tmp_path / "index.json",
        )


def test_load_index_restores_saved_data(tmp_path: Path) -> None:
    """A saved index should be loadable without data loss."""
    index_path = tmp_path / "index.json"

    save_index(
        chunks=sample_chunks(),
        embeddings=sample_embeddings(),
        model="text-embedding-3-small",
        path=index_path,
    )

    index = load_index(index_path)

    assert index["embedding_model"] == "text-embedding-3-small"
    assert len(index["chunks"]) == 2
    assert index["chunks"][0]["chunk_id"] == "chunk_001"


def test_load_index_raises_when_file_is_missing(
    tmp_path: Path,
) -> None:
    """Loading a missing index should produce a clear error."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Index not found"):
        load_index(missing_path)


def test_load_index_rejects_invalid_json(tmp_path: Path) -> None:
    """A malformed index file should produce a clear error."""
    index_path = tmp_path / "index.json"
    index_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        load_index(index_path)