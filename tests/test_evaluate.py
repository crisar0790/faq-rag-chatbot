import json
from types import SimpleNamespace

import pytest
from pathlib import Path
from unittest.mock import Mock

from src.evaluate import (
    _evaluate_retrieval,
    build_evaluation_input,
    build_summary,
    evaluate_answer,
    evaluate_rag_output,
    main,
    validate_evaluation_result,
)

from argparse import Namespace

from src.constants import EVALUATOR_MAX_OUTPUT_TOKENS

def test_build_evaluation_input():
    result = build_evaluation_input(
        question="How can I reset my password?",
        answer="Use the password reset option.",
        chunks=[
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "Users can reset their password.",
            }
        ],
    )

    assert "How can I reset my password?" in result
    assert "Use the password reset option." in result
    assert "chunk_001" in result
    assert "Users can reset their password." in result


def test_build_summary():
    results = [
        {
            "retrieval_rank": 1,
            "retrieval_passed": True,
            "section_precision_at_k": 0.6667,
            "answer_passed": True,
            "passed": True,
        },
        {
            "retrieval_rank": 2,
            "retrieval_passed": True,
            "section_precision_at_k": 0.3333,
            "answer_passed": False,
            "passed": False,
        },
        {
            "retrieval_rank": None,
            "retrieval_passed": False,
            "section_precision_at_k": 0.0,
            "answer_passed": True,
            "passed": False,
        },
    ]

    summary = build_summary(results)

    assert summary == {
        "total_cases": 3,
        "retrieval_top_1_passed": 1,
        "retrieval_top_1_accuracy": 0.3333,
        "retrieval_passed": 2,
        "retrieval_accuracy": 0.6667,
        "mean_section_precision_at_k": 0.3333,
        "answer_passed": 2,
        "answer_accuracy": 0.6667,
        "fully_passed": 1,
        "overall_accuracy": 0.3333,
    }

def test_evaluate_retrieval_calculates_precision_at_k():
    evaluation_case = {
        "id": "eval_001",
        "question": "How can I reset my password?",
        "expected_section": "Account Access",
    }
    chunks = [
        {
            "section": "Account Access",
        },
        {
            "section": "Account Access",
        },
        {
            "section": "Data Privacy",
        },
    ]

    result = _evaluate_retrieval(
        evaluation_case,
        chunks,
    )

    assert result == (
        [
            "Account Access",
            "Data Privacy",
        ],
        1,
        2,
        0.6667,
    )

class FakeResponsesAPI:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponsesAPI(response)


def test_validate_evaluation_result():
    evaluation = validate_evaluation_result(
        {
            "score": 8,
            "reason": (
                "The answer is grounded in chunk_001, "
                "directly addresses the question, and "
                "includes the available relevant details."
            ),
        }
    )

    assert evaluation["score"] == 8
    assert len(evaluation["reason"]) >= 50


@pytest.mark.parametrize("score", [-1, 11, 7.5, True])
def test_rejects_invalid_evaluation_score(score):
    with pytest.raises(
        ValueError,
        match="score must be an integer",
    ):
        validate_evaluation_result(
            {
                "score": score,
                "reason": (
                    "This explanation is intentionally "
                    "long enough to pass validation."
                ),
            }
        )


def test_rejects_short_evaluation_reason():
    with pytest.raises(
        ValueError,
        match="at least 50 characters",
    ):
        validate_evaluation_result(
            {
                "score": 8,
                "reason": "Too short.",
            }
        )


def test_evaluate_answer_returns_score_and_reason():
    response = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {
                "score": 9,
                "reason": (
                    "The answer is fully grounded in "
                    "chunk_001, directly answers the "
                    "question, and contains all relevant "
                    "information from the context."
                ),
            }
        ),
    )
    client = FakeClient(response)

    evaluation = evaluate_answer(
        question="How can I reset my password?",
        answer="Use the password reset option.",
        chunks=[
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": (
                    "Employees can use the password "
                    "reset option."
                ),
            }
        ],
        client=client,
        model="gpt-4o-mini",
    )

    assert evaluation["score"] == 9
    assert "chunk_001" in evaluation["reason"]

    request = client.responses.last_request
    schema = request["text"]["format"]["schema"]

    assert (
        request["max_output_tokens"]
        == EVALUATOR_MAX_OUTPUT_TOKENS
    )

    assert schema["properties"]["score"]["minimum"] == 0
    assert schema["properties"]["score"]["maximum"] == 10
    assert schema["properties"]["reason"]["minLength"] == 50


def test_evaluate_rag_output():
    response = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {
                "score": 8,
                "reason": (
                    "The answer is supported by chunk_001 "
                    "and is relevant, although it could "
                    "include one additional contextual "
                    "detail."
                ),
            }
        ),
    )
    client = FakeClient(response)

    result = evaluate_rag_output(
        rag_output={
            "user_question": (
                "How can I reset my password?"
            ),
            "system_answer": (
                "Use the password reset option."
            ),
            "chunks_related": [
                {
                    "chunk_id": "chunk_001",
                    "section": "Account Access",
                    "text": (
                        "Employees can use the password "
                        "reset option."
                    ),
                }
            ],
        },
        client=client,
        model="gpt-4o-mini",
    )

    assert set(result) == {"score", "reason"}
    assert result["score"] == 8


def test_rejects_invalid_rag_output():
    with pytest.raises(
        ValueError,
        match="must contain exactly",
    ):
        evaluate_rag_output(
            rag_output={
                "user_question": "Test question",
                "system_answer": "Test answer",
            },
            client=object(),
            model="gpt-4o-mini",
        )

def test_evaluation_main_formats_errors(
    monkeypatch,
    capsys,
):
    args = Namespace(
        dataset=Path("questions.json"),
        index=Path("index.json"),
        report=Path("report.json"),
        top_k=3,
        limit=None,
    )

    monkeypatch.setattr(
        "src.evaluate.parse_arguments",
        lambda: args,
    )
    monkeypatch.setattr(
        "src.evaluate.run_evaluation",
        Mock(
            side_effect=ValueError(
                "Invalid evaluation configuration."
            )
        ),
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 1

    captured = capsys.readouterr()

    assert "Invalid evaluation configuration." in (
        captured.err
    )