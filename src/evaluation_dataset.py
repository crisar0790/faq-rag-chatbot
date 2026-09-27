"""Load and validate the dataset used to evaluate the RAG system."""

import json
from pathlib import Path
from typing import TypedDict


DEFAULT_DATASET_PATH = Path(
    "evaluation/questions.json"
)


class EvaluationCase(TypedDict):
    """Represent one question and its expected document section."""
    id: str
    question: str
    expected_section: str


def validate_evaluation_case(evaluation_case: object, position: int) -> EvaluationCase:
    """Validate and normalize one evaluation dataset entry."""
    if not isinstance(evaluation_case, dict):
        raise ValueError(
            f"Evaluation case {position} must be an object."
        )

    expected_fields = {
        "id",
        "question",
        "expected_section",
    }

    if set(evaluation_case) != expected_fields:
        raise ValueError(
            f"Evaluation case {position} must contain "
            "exactly: id, question and expected_section."
        )

    validated_case = {}

    for field in expected_fields:
        value = evaluation_case[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Evaluation case {position} has an "
                f"invalid '{field}' value."
            )

        validated_case[field] = value.strip()

    return EvaluationCase(
        id=validated_case["id"],
        question=validated_case["question"],
        expected_section=(
            validated_case["expected_section"]
        ),
    )


def load_evaluation_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[EvaluationCase]:
    """Load a non-empty evaluation dataset with unique case identifiers."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Evaluation dataset contains invalid JSON."
        ) from error

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Evaluation dataset must be a non-empty list."
        )

    evaluation_cases = [
        validate_evaluation_case(
            evaluation_case=item,
            position=position,
        )
        for position, item in enumerate(
            data,
            start=1,
        )
    ]

    case_ids = [
        evaluation_case["id"]
        for evaluation_case in evaluation_cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Evaluation case IDs must be unique."
        )

    return evaluation_cases