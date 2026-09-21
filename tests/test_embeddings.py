"""Tests for embedding generation."""

from types import SimpleNamespace

import pytest

from src.embeddings import generate_embeddings


class FakeEmbeddingsAPI:
    """Simulate the OpenAI embeddings API."""

    def __init__(self) -> None:
        """Initialize the fake API and its call history."""
        self.calls: list[dict] = []

    def create(self, *, input: list[str], model: str):
        """Return deterministic vectors without calling OpenAI."""
        self.calls.append(
            {
                "input": input,
                "model": model,
            }
        )

        data = [
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

        return SimpleNamespace(data=data)


class FakeOpenAIClient:
    """Provide the API surface required by the embedding service."""

    def __init__(self) -> None:
        """Create a fake embeddings endpoint."""
        self.embeddings = FakeEmbeddingsAPI()

class IncompleteEmbeddingsAPI:
    """Return fewer embeddings than requested."""

    def create(self, *, input: list[str], model: str):
        """Simulate an incomplete embeddings response."""
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 2.0, 3.0],
                )
            ]
        )


class IncompleteOpenAIClient:
    """Provide an incomplete embeddings endpoint."""

    def __init__(self) -> None:
        """Create the incomplete fake endpoint."""
        self.embeddings = IncompleteEmbeddingsAPI()


class InconsistentEmbeddingsAPI:
    """Return vectors with different dimensions."""

    def create(self, *, input: list[str], model: str):
        """Simulate inconsistent embedding dimensions."""
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 2.0, 3.0],
                ),
                SimpleNamespace(
                    index=1,
                    embedding=[1.0, 2.0],
                ),
            ]
        )


class InconsistentOpenAIClient:
    """Provide an inconsistent embeddings endpoint."""

    def __init__(self) -> None:
        """Create the inconsistent fake endpoint."""
        self.embeddings = InconsistentEmbeddingsAPI()

def test_generate_embeddings_calls_api_with_texts_and_model() -> None:
    """The service should send every text to the configured model."""
    client = FakeOpenAIClient()
    texts = [
        "First AR HR support chunk.",
        "Second AR HR support chunk.",
    ]

    generate_embeddings(
        texts=texts,
        client=client,
        model="text-embedding-3-small",
    )

    assert client.embeddings.calls == [
        {
            "input": texts,
            "model": "text-embedding-3-small",
        }
    ]


def test_generate_embeddings_returns_vectors_in_source_order() -> None:
    """Generated vectors should preserve the order of their texts."""
    client = FakeOpenAIClient()
    texts = ["First chunk.", "Second chunk."]

    embeddings = generate_embeddings(
        texts=texts,
        client=client,
        model="text-embedding-3-small",
    )

    assert embeddings == [
        [1.0, float(len(texts[0])), 1.0],
        [2.0, float(len(texts[1])), 1.0],
    ]


def test_generate_embeddings_rejects_empty_input() -> None:
    """At least one text is required to generate embeddings."""
    client = FakeOpenAIClient()

    with pytest.raises(ValueError, match="at least one text"):
        generate_embeddings(
            texts=[],
            client=client,
            model="text-embedding-3-small",
        )


def test_generate_embeddings_rejects_blank_text() -> None:
    """Blank chunks should not be sent to the embeddings API."""
    client = FakeOpenAIClient()

    with pytest.raises(ValueError, match="must not be blank"):
        generate_embeddings(
            texts=["Valid chunk.", "   "],
            client=client,
            model="text-embedding-3-small",
        )


def test_generate_embeddings_rejects_empty_model() -> None:
    """An embedding model name is required."""
    client = FakeOpenAIClient()

    with pytest.raises(ValueError, match="model"):
        generate_embeddings(
            texts=["Valid chunk."],
            client=client,
            model="",
        )

def test_generate_embeddings_rejects_incomplete_response() -> None:
    """The API must return one embedding for every input text."""
    client = IncompleteOpenAIClient()

    with pytest.raises(ValueError, match="number of embeddings"):
        generate_embeddings(
            texts=["First chunk.", "Second chunk."],
            client=client,
            model="text-embedding-3-small",
        )


def test_generate_embeddings_rejects_inconsistent_dimensions() -> None:
    """Every returned embedding should have the same dimension."""
    client = InconsistentOpenAIClient()

    with pytest.raises(ValueError, match="dimension"):
        generate_embeddings(
            texts=["First chunk.", "Second chunk."],
            client=client,
            model="text-embedding-3-small",
        )