"""Generate vector embedding for document chunks."""

from typing import Protocol

class EmbeddingRecord(Protocol):
    """Describe one embedding returned by the API."""

    index: int
    embedding: list[float]

class EmbeddingResponse(Protocol):
    """Describe the relevant part of an embeddings response."""

    data: list[EmbeddingRecord]

class EmbeddingsAPI(Protocol):
    """Describe the embeddings endpoint used by the service."""

    def create(self, *, input: list[str], model: str) -> EmbeddingResponse:
        """Generate embeddings for the provided texts."""

class OpenAIClient(Protocol):
    """Describe the required OpenAI client interface."""

    embeddings: EmbeddingsAPI

def validate_embedding_input(texts: list[str], model: str) -> None:
    """Validate texts and model before calling the API."""
    if not texts:
        raise ValueError(
            "Embedding generation requires at least one text."
        )

    if any(not text.strip() for text in texts):
        raise ValueError("Embedding texts must not be blank.")

    if not model.strip():
        raise ValueError("The embedding model must not be empty.")

def validate_embedding_response(embeddings: list[list[float]], expected_count: int) -> None:
    """Validate the amount and dimensions of embedding vectors."""
    if len(embeddings) != expected_count:
        raise ValueError(
            "The number of embeddings does not match "
            "the number of input texts."
        )

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError(
            "All embedding vectors must have the same dimension."
        )

def generate_embeddings(texts: list[str], client: OpenAIClient, model: str) -> list[list[float]]:
    """Generate one embedding vector for each input text."""
    validate_embedding_input(texts, model)

    response = client.embeddings.create(
        input=texts,
        model=model,
    )

    ordered_records = sorted(
        response.data,
        key=lambda record: record.index,
    )
    embeddings = [
        record.embedding
        for record in ordered_records
    ]

    validate_embedding_response(
        embeddings=embeddings,
        expected_count=len(texts),
    )

    return embeddings