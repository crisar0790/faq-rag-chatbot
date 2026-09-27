"""Generate reproducible sample outputs from the RAG pipeline."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import SAMPLE_OUTPUT_PATH
from src.query import run_query


OUTPUT_PATH = SAMPLE_OUTPUT_PATH

SAMPLE_QUESTIONS = [
    "How can I reset my password?",
    "How should an employee submit a work expense?",
    "How can a customer contact AR HR support?",
]


def generate_samples(questions: list[str], query_function: Callable[[str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the configured questions through the RAG pipeline."""
    return [
        query_function(question)
        for question in questions
    ]


def save_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Save sample RAG outputs as formatted JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            samples,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Generate and persist the required sample query outputs."""
    samples = generate_samples(
        questions=SAMPLE_QUESTIONS,
        query_function=run_query,
    )
    save_samples(
        samples=samples,
        output_path=OUTPUT_PATH,
    )

    print(
        f"Saved {len(samples)} sample queries "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()