"""Tests for the end-to-end indexing pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.build_index import build_index
from src.config import DOCUMENT_PATH


class FakeEmbeddingsAPI:
    """Simulate deterministic embedding generation."""

    def __init__(self) -> None:
        """Initialize the fake call history."""
        self.calls: list[dict] = []

    def create(self, *, input: list[str], model: str):
        """Return one deterministic vector per text."""
        self.calls.append(
            {
                "input": input,
                "model": model,
            }
        )

        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=index,
                    embedding=[
                        float(index + 1),
                        float(len(text)),
                        1.0,
                    ],
                )
                for index, text in enumerate(input)
            ]
        )


class FakeOpenAIClient:
    """Provide the API used by the indexing pipeline."""

    def __init__(self) -> None:
        """Create the fake embeddings endpoint."""
        self.embeddings = FakeEmbeddingsAPI()


def test_build_index_creates_complete_vector_index(
    tmp_path: Path,
) -> None:
    """The pipeline should process and persist the entire document."""
    client = FakeOpenAIClient()
    index_path = tmp_path / "index.json"

    summary = build_index(
        client=client,
        document_path=DOCUMENT_PATH,
        index_path=index_path,
        model="text-embedding-3-small",
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index_path.is_file()
    assert len(index["chunks"]) >= 20
    assert summary["chunk_count"] == len(index["chunks"])
    assert summary["embedding_count"] == len(index["chunks"])
    assert summary["embedding_dimension"] == 3


def test_build_index_sends_every_chunk_to_embeddings_api(
    tmp_path: Path,
) -> None:
    """Every generated chunk should be sent for embedding."""
    client = FakeOpenAIClient()

    build_index(
        client=client,
        document_path=DOCUMENT_PATH,
        index_path=tmp_path / "index.json",
        model="text-embedding-3-small",
    )

    assert len(client.embeddings.calls) == 1

    call = client.embeddings.calls[0]

    assert call["model"] == "text-embedding-3-small"
    assert len(call["input"]) >= 20
    assert all(text.strip() for text in call["input"])