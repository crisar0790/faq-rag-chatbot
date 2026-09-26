import json
from pathlib import Path

import pytest

from src.evaluation_dataset import (
    load_evaluation_dataset,
)


def write_dataset(
    path: Path,
    data: object,
) -> None:
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def test_loads_valid_evaluation_dataset(
    tmp_path,
):
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "id": "eval_001",
                "question": "How can I reset my password?",
                "expected_section": (
                    "Account Access and Password Recovery"
                ),
            }
        ],
    )

    dataset = load_evaluation_dataset(dataset_path)

    assert len(dataset) == 1
    assert dataset[0]["id"] == "eval_001"
    assert dataset[0]["question"] == (
        "How can I reset my password?"
    )


def test_rejects_empty_dataset(tmp_path):
    dataset_path = tmp_path / "questions.json"
    write_dataset(dataset_path, [])

    with pytest.raises(
        ValueError,
        match="non-empty list",
    ):
        load_evaluation_dataset(dataset_path)


def test_rejects_missing_field(tmp_path):
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "id": "eval_001",
                "question": "How can I reset my password?",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="must contain exactly",
    ):
        load_evaluation_dataset(dataset_path)


def test_rejects_empty_field(tmp_path):
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "id": "eval_001",
                "question": "   ",
                "expected_section": "Account Access",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="invalid 'question'",
    ):
        load_evaluation_dataset(dataset_path)


def test_rejects_duplicate_ids(tmp_path):
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "id": "eval_001",
                "question": "First question",
                "expected_section": "First Section",
            },
            {
                "id": "eval_001",
                "question": "Second question",
                "expected_section": "Second Section",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="IDs must be unique",
    ):
        load_evaluation_dataset(dataset_path)


def test_rejects_invalid_json(tmp_path):
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        "{invalid JSON}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        load_evaluation_dataset(dataset_path)