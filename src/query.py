"""Execute individual questions through the complete RAG pipeline."""

import argparse
import json
from pathlib import Path
from typing import Any
import sys

from src.config import (
    get_embedding_model,
    get_llm_model,
    get_max_output_tokens,
    get_openai_client,
)
from src.embeddings import generate_embeddings
from src.rag_service import answer_question
from src.retrieval import search_similar_chunks
from src.vector_store import load_index
from src.errors import format_error

DEFAULT_INDEX_PATH = Path("data/index.json")
DEFAULT_TOP_K = 3

def create_query_embedding(text: str, client: Any, model: str) -> list[float]:
    """Generate a single embedding vector for a user question."""
    embeddings = generate_embeddings(texts=[text], client=client, model=model)

    return embeddings[0]

def retrieve_chunks(query_embedding: list[float], index: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Retrieve the most similar document chunks for a query vector."""
    return search_similar_chunks(index=index, query_embedding=query_embedding, top_k=top_k)

def run_query(question: str, index_path: Path = DEFAULT_INDEX_PATH, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Run one question through retrieval and grounded answer generation."""
    index = load_index(index_path)
    client = get_openai_client()

    return answer_question(
        question=question,
        index=index,
        client=client,
        embedding_model=get_embedding_model(),
        llm_model=get_llm_model(),
        max_output_tokens=get_max_output_tokens(),
        create_query_embedding=create_query_embedding,
        retrieve_chunks=retrieve_chunks,
        top_k=top_k,
    )

def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for individual queries."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask a question using the AR HR FAQ "
            "knowledge base."
        )
    )
    parser.add_argument(
        "question",
        help="Question to answer using the FAQ document.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the vector index. Default: data/index.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve. Default: 3",
    )

    return parser

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for an individual FAQ query."""
    return _build_argument_parser().parse_args()

def main() -> None:
    """Execute the command-line query and print its JSON result."""
    args = parse_arguments()

    try:
        result = run_query(
            question=args.question,
            index_path=args.index,
            top_k=args.top_k,
        )
    except Exception as error:
        print(
            format_error(error),
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    main()