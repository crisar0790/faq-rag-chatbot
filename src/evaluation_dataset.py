"""Load and validate the dataset used to evaluate the RAG system."""

import json
from pathlib import Path
from typing import Any, TypedDict
from src.config import EVALUATION_DATASET_PATH


DEFAULT_DATASET_PATH = EVALUATION_DATASET_PATH

EXPECTED_FIELDS = {
    "id",
    "question",
    "expected_section",
}


class EvaluationCase(TypedDict):
    """Represent one question and its expected document section."""

    id: str
    question: str
    expected_section: str


def _read_dataset(path: Path) -> Any:
    """Read and decode an evaluation dataset file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Evaluation dataset contains invalid JSON."
        ) from error


def _validate_text_field(evaluation_case: dict[str, Any], field: str, position: int) -> str:
    """Validate and normalize one text field."""
    value = evaluation_case[field]

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Evaluation case {position} has an "
            f"invalid '{field}' value."
        )

    return value.strip()


def validate_evaluation_case(evaluation_case: object, position: int) -> EvaluationCase:
    """Validate and normalize one evaluation dataset entry."""
    if not isinstance(evaluation_case, dict):
        raise ValueError(
            f"Evaluation case {position} must be an object."
        )

    if set(evaluation_case) != EXPECTED_FIELDS:
        raise ValueError(
            f"Evaluation case {position} must contain "
            "exactly: id, question and expected_section."
        )

    return EvaluationCase(
        id=_validate_text_field(
            evaluation_case, "id", position
        ),
        question=_validate_text_field(
            evaluation_case, "question", position
        ),
        expected_section=_validate_text_field(
            evaluation_case, "expected_section", position
        ),
    )


def _validate_unique_ids(evaluation_cases: list[EvaluationCase]) -> None:
    """Ensure every evaluation case has a unique identifier."""
    case_ids = [
        evaluation_case["id"]
        for evaluation_case in evaluation_cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Evaluation case IDs must be unique."
        )


def load_evaluation_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[EvaluationCase]:
    """Load a non-empty dataset with unique case identifiers."""
    data = _read_dataset(path)

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Evaluation dataset must be a non-empty list."
        )

    evaluation_cases = [
        validate_evaluation_case(item, position)
        for position, item in enumerate(data, start=1)
    ]

    _validate_unique_ids(evaluation_cases)

    return evaluation_cases