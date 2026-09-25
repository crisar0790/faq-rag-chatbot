"""Build and validate the required RAG response contract."""

from typing import Any, TypedDict

MIN_RELATED_CHUNKS = 2
MAX_RELATED_CHUNKS = 5

REQUIRED_OUTPUT_KEYS = {
    "user_question",
    "system_answer",
    "chunks_related",
}

REQUIRED_CHUNK_KEYS = {
    "chunk_id",
    "section",
    "text",
}

class RelatedChunk(TypedDict):
    """Public source information included in a RAG result."""

    chunk_id: str
    section: str
    text: str

class RAGOutput(TypedDict):
    """Required public output returned by the RAG system."""

    user_question: str
    system_answer: str
    chunks_related: list[RelatedChunk]

def validate_nonblank_text(value: Any, field_name: str) -> None:
    """Ensure a value is a nonblank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must not be blank."
        )

def validate_related_chunks(chunks: Any) -> None:
    """Validate the public related chunk collection."""
    if not isinstance(chunks, list):
        raise ValueError(
            "chunks_related must be a list."
        )

    if not MIN_RELATED_CHUNKS <= len(chunks) <= MAX_RELATED_CHUNKS:
        raise ValueError(
            "chunks_related must contain between 2 and 5 chunks."
        )

    chunk_ids: list[str] = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(
                "Every related chunk must be an object."
            )

        if set(chunk) != REQUIRED_CHUNK_KEYS:
            raise ValueError(
                "Each related chunk must contain exactly "
                "chunk_id, section, and text."
            )

        for key in REQUIRED_CHUNK_KEYS:
            validate_nonblank_text(chunk[key], key)

        chunk_ids.append(chunk["chunk_id"])

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Related chunk identifiers must be unique."
        )


def validate_rag_output(output: Any) -> None:
    """Validate the complete public RAG output."""
    if not isinstance(output, dict):
        raise ValueError("RAG output must be an object.")

    if set(output) != REQUIRED_OUTPUT_KEYS:
        raise ValueError(
            "RAG output must contain exactly "
            "user_question, system_answer, and chunks_related."
        )

    validate_nonblank_text(
        output["user_question"],
        "user_question",
    )
    validate_nonblank_text(
        output["system_answer"],
        "system_answer",
    )
    validate_related_chunks(
        output["chunks_related"],
    )


def build_public_chunks(chunks: list[dict[str, Any]]) -> list[RelatedChunk]:
    """Remove internal fields from retrieved chunks."""
    return [
        RelatedChunk(
            chunk_id=chunk["chunk_id"],
            section=chunk["section"],
            text=chunk["text"],
        )
        for chunk in chunks
    ]


def build_rag_output(question: str, answer: str, chunks: list[dict[str, Any]]) -> RAGOutput:
    """Build and validate the required RAG result."""
    output = RAGOutput(
        user_question=question.strip(),
        system_answer=answer.strip(),
        chunks_related=build_public_chunks(chunks),
    )

    validate_rag_output(output)
    return output