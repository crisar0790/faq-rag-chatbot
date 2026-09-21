"""Persist and load document chunks with their embeddings."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

INDEX_VERSION = 1

def validate_index_input(chunks: list[dict], embeddings: list[list[float]], model: str) -> int:
    """Validate index data and return its embedding dimension."""
    if not chunks:
        raise ValueError("The index requires at least one chunk.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of embeddings must match "
            "the number of chunks."
        )

    if not model.strip():
        raise ValueError("The embedding must not be empty.")

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("All embeddings mut have the same dimension.")

    return dimensions.pop()

def build_index_data(chunks: list[dict], embeddings: list[list[float]], model: str) -> dict[str, Any]:
    """Combine chunks, embeddings, and index metadata."""
    dimension = validate_index_input(chunks=chunks, embeddings=embeddings, model=model)

    stored_chunks = [
        {
            **chunk,
            "embedding": embedding
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    return {
        "index_version": INDEX_VERSION,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "chunks": stored_chunks,
    }

def save_index(chunks: list[dict], embeddings: list[list[float]], model: str, path: Path) -> None:
    """Save chunks and embeddings as a JSON vector index."""
    index = build_index_data(chunks=chunks, embeddings=embeddings, model=model)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            index,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

def load_index(path: Path) -> dict[str, Any]:
    """Load a previously generated JSON vector index."""
    if not path.is_file():
        raise FileNotFoundError(f"Index not found: {path}")

    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise ValueError(
            f"Index must contain valid JSON: {path}"
        ) from error

    return index
