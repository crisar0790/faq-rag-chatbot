"""Build the persistent vector index from the support document."""

from pathlib import Path
from typing import Any

from src.chunker import chunk_document
from src.config import (
    DOCUMENT_PATH,
    INDEX_PATH,
    get_embedding_model,
    get_openai_client,
)
from src.document_loader import load_document
from src.embeddings import OpenAIClient, generate_embeddings
from src.vector_store import save_index

def build_index(client: OpenAIClient, document_path: Path, index_path: Path, model: str) -> dict[str, Any]:
    """Run the complete document indexing pipeline."""
    document = load_document(document_path)
    chunks = chunk_document(document)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        texts=texts,
        client=client,
        model=model,
    )

    save_index(
        chunks=chunks,
        embeddings=embeddings,
        model=model,
        path=index_path,
    )

    return {
        "word_count": len(document.split()),
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "index_path": str(index_path),
    }

def print_summary(summary: dict[str, Any]) -> None:
    """Print a readable indexing summary."""
    print("Index built successfully")
    print(f'Words processed: {summary["word_count"]}')
    print(f'Chuns created: {summary["chunk_count"]}')
    print(f'Embeddings created: {summary["embedding_count"]}')
    print(
        "Embedding dimension: "
        f'{summary["embedding_dimension"]}'
    )
    print(f'Index saved to: {summary["index_path"]}')

def main() -> None:
    """Build the real vector index using OpenAI."""
    model = get_embedding_model()
    client = get_openai_client()

    summary = build_index(
        client=client,
        document_path=DOCUMENT_PATH,
        index_path=INDEX_PATH,
        model=model,
    )

    print_summary(summary)

if __name__ == "__main__":
    main()