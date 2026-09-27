"""Persist and load document chunks with their embeddings."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any
import math

INDEX_VERSION = 1

REQUIRED_INDEX_KEYS = {
    "index_version",
    "embedding_model",
    "embedding_dimension",
    "chunks",
}

REQUIRED_STORED_CHUNK_KEYS = {
    "chunk_id",
    "section",
    "text",
    "token_count",
    "embedding",
}

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
        raise ValueError(
            "The embedding model must not be empty."
        )

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("All embeddings must have the same dimension.")

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

def _validate_index_metadata(index: Any) -> tuple[int, list[Any]]:
    """Validate index metadata and return its dimension and chunks."""
    if not isinstance(index, dict):
        raise ValueError("The vector index must be a JSON object.")

    if set(index) != REQUIRED_INDEX_KEYS:
        raise ValueError("The vector index has missing or unexpected fields.")

    if index["index_version"] != INDEX_VERSION:
        raise ValueError("The vector index version is not supported.")

    if (
        not isinstance(index["embedding_model"], str)
        or not index["embedding_model"].strip()
    ):
        raise ValueError("The vector index has an invalid embedding model.")

    dimension = index["embedding_dimension"]
    chunks = index["chunks"]

    if type(dimension) is not int or dimension <= 0:
        raise ValueError("The vector index has an invalid embedding dimension.")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The vector index must contain at least one chunk.")

    return dimension, chunks

def _validate_stored_embedding(
    embedding: Any,
    expected_dimension: int,
) -> None:
    """Validate one stored embedding vector."""
    if (
        not isinstance(embedding, list)
        or len(embedding) != expected_dimension
    ):
        raise ValueError(
            "A stored embedding has an invalid dimension."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in embedding
    ):
        raise ValueError(
            "Stored embeddings must contain finite numbers."
        )

def _validate_stored_chunk(chunk: Any, expected_dimension: int) -> str:
    """Validate one stored chunk and return its identifier."""
    if not isinstance(chunk, dict):
        raise ValueError(
            "Every stored chunk must be an object."
        )

    if set(chunk) != REQUIRED_STORED_CHUNK_KEYS:
        raise ValueError(
            "A stored chunk has missing or unexpected fields."
        )

    for field in ("chunk_id", "section", "text"):
        value = chunk[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Stored chunk field '{field}' must not be blank."
            )

    token_count = chunk["token_count"]

    if type(token_count) is not int or token_count <= 0:
        raise ValueError(
            "Stored chunk token_count must be a positive integer."
        )

    _validate_stored_embedding(
        chunk["embedding"],
        expected_dimension,
    )

    return chunk["chunk_id"]

def validate_loaded_index(
    index: Any,
) -> None:
    """Validate the complete structure of a loaded vector index."""
    dimension, chunks = _validate_index_metadata(index)

    chunk_ids = [
        _validate_stored_chunk(chunk, dimension)
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Stored chunk identifiers must be unique."
        )

def load_index(path: Path) -> dict[str, Any]:
    """Load and validate a previously generated JSON index."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Index not found: {path}"
        )

    try:
        index = json.loads(
            path.read_text(encoding="utf-8")
        )
    except JSONDecodeError as error:
        raise ValueError(
            f"Index must contain valid JSON: {path}"
        ) from error

    validate_loaded_index(index)

    return index

def validate_index_model(
    index: dict[str, Any],
    configured_model: str,
) -> None:
    """Ensure the index uses the configured embedding model."""
    indexed_model = index.get("embedding_model")

    if not isinstance(indexed_model, str):
        raise ValueError(
            "The vector index does not contain a valid "
            "embedding model."
        )

    if indexed_model != configured_model:
        raise ValueError(
            "The vector index was generated with "
            f"'{indexed_model}', but the configured model "
            f"is '{configured_model}'. Rebuild the index."
        )