"""Evaluate retrieval and answer quality across a question dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import (
    EVALUATION_REPORT_PATH,
    EVALUATOR_PROMPT_PATH,
    get_embedding_model,
    get_llm_model,
    get_max_output_tokens,
    get_openai_client,
)
from src.evaluation_dataset import (
    DEFAULT_DATASET_PATH,
    load_evaluation_dataset,
)
from src.query import (
    DEFAULT_INDEX_PATH,
    create_query_embedding,
    retrieve_chunks,
)
from src.rag_service import answer_question
from src.vector_store import (
    load_index,
    validate_index_model,
)


DEFAULT_REPORT_PATH = EVALUATION_REPORT_PATH

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
        },
        "reason": {
            "type": "string",
            "minLength": 50,
        },
    },
    "required": [
        "score",
        "reason",
    ],
    "additionalProperties": False,
}

PASSING_SCORE = 7
MIN_REASON_LENGTH = 50

def load_evaluator_prompt(path: Path = EVALUATOR_PROMPT_PATH) -> str:
    """Load and validate the prompt used by the answer evaluator."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluator prompt not found: {path}"
        )

    prompt = path.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt:
        raise ValueError(
            "Evaluator prompt cannot be empty."
        )

    return prompt


def build_evaluation_input(question: str, answer: str, chunks: list[dict[str, Any]]) -> str:
    """Build the question, context, and answer input for evaluation."""
    context = "\n\n".join(
        (
            f"[{chunk['chunk_id']} | "
            f"{chunk['section']}]\n"
            f"{chunk['text']}"
        )
        for chunk in chunks
    )

    return (
        "USER QUESTION\n"
        "-------------\n"
        f"{question}\n\n"
        "RETRIEVED CONTEXT\n"
        "-----------------\n"
        f"{context}\n\n"
        "SYSTEM ANSWER\n"
        "-------------\n"
        f"{answer}"
    )

def _validate_evaluation_score(score: Any) -> int:
    """Validate and return an evaluator score."""
    if type(score) is not int or not 0 <= score <= 10:
        raise ValueError(
            "The evaluation score must be an integer "
            "between 0 and 10."
        )

    return score


def _validate_evaluation_reason(reason: Any) -> str:
    """Validate and normalize an evaluator reason."""
    if not isinstance(reason, str):
        raise ValueError(
            "The evaluation reason must be a string."
        )

    clean_reason = reason.strip()

    if len(clean_reason) < MIN_REASON_LENGTH:
        raise ValueError(
            "The evaluation reason must contain at "
            "least 50 characters."
        )

    return clean_reason

def validate_evaluation_result(evaluation: Any) -> dict[str, Any]:
    """Validate the score and reason returned by the evaluator."""
    if not isinstance(evaluation, dict):
        raise ValueError(
            "The evaluation result must be an object."
        )

    if set(evaluation) != {"score", "reason"}:
        raise ValueError(
            "The evaluation result must contain exactly "
            "score and reason."
        )

    return {
        "score": _validate_evaluation_score(
            evaluation["score"]
        ),
        "reason": _validate_evaluation_reason(
            evaluation["reason"]
        ),
    }

def _build_evaluation_messages(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the system and user messages for evaluation."""
    return [
        {
            "role": "system",
            "content": load_evaluator_prompt(),
        },
        {
            "role": "user",
            "content": build_evaluation_input(
                question,
                answer,
                chunks,
            ),
        },
    ]

def _request_evaluation(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    client: Any,
    model: str,
) -> Any:
    """Request a structured evaluation from OpenAI."""
    return client.responses.create(
        model=model,
        input=_build_evaluation_messages(
            question,
            answer,
            chunks,
        ),
        max_output_tokens=300,
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_evaluation",
                "strict": True,
                "schema": EVALUATION_SCHEMA,
            }
        },
    )


def _parse_evaluation_response(response: Any) -> dict[str, Any]:
    """Parse and validate a structured evaluation response."""
    if response.status != "completed":
        raise ValueError(
            "The evaluator response was not completed."
        )

    try:
        evaluation = json.loads(
            response.output_text
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "The evaluator returned invalid JSON."
        ) from error

    return validate_evaluation_result(evaluation)

def evaluate_answer(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    client: Any,
    model: str,
) -> dict[str, Any]:
    """Evaluate an answer using the configured scoring rubric."""
    response = _request_evaluation(
        question=question,
        answer=answer,
        chunks=chunks,
        client=client,
        model=model,
    )

    return _parse_evaluation_response(response)

def evaluate_rag_output(rag_output: dict[str, Any], client: Any, model: str) -> dict[str, Any]:
    """Evaluate one complete public RAG response."""
    required_fields = {
        "user_question",
        "system_answer",
        "chunks_related",
    }

    if set(rag_output) != required_fields:
        raise ValueError(
            "The RAG output must contain exactly "
            "user_question, system_answer and "
            "chunks_related."
        )

    return evaluate_answer(
        question=rag_output["user_question"],
        answer=rag_output["system_answer"],
        chunks=rag_output["chunks_related"],
        client=client,
        model=model,
    )

def _run_evaluation_query(
    evaluation_case: dict[str, str],
    index: dict[str, Any],
    client: Any,
    embedding_model: str,
    llm_model: str,
    max_output_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    """Run one dataset question through the RAG pipeline."""
    return answer_question(
        question=evaluation_case["question"],
        index=index,
        client=client,
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_output_tokens=max_output_tokens,
        create_query_embedding=create_query_embedding,
        retrieve_chunks=retrieve_chunks,
        top_k=top_k,
    )


def _get_retrieved_sections(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """Return unique retrieved section names in result order."""
    return list(
        dict.fromkeys(
            chunk["section"]
            for chunk in chunks
        )
    )


def _build_case_result(
    evaluation_case: dict[str, str],
    rag_output: dict[str, Any],
    retrieved_sections: list[str],
    retrieval_passed: bool,
    answer_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Build the persisted result for one evaluation case."""
    answer_passed = (
        answer_evaluation["score"] >= PASSING_SCORE
    )

    return {
        "id": evaluation_case["id"],
        "question": evaluation_case["question"],
        "expected_section": evaluation_case["expected_section"],
        "retrieved_sections": retrieved_sections,
        "retrieval_passed": retrieval_passed,
        "system_answer": rag_output["system_answer"],
        "answer_evaluation": answer_evaluation,
        "answer_passed": answer_passed,
        "passed": retrieval_passed and answer_passed,
    }

def _evaluate_retrieval(
    evaluation_case: dict[str, str],
    chunks: list[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Return retrieved sections and whether the expected one appears."""
    retrieved_sections = _get_retrieved_sections(
        chunks
    )
    retrieval_passed = (
        evaluation_case["expected_section"]
        in retrieved_sections
    )

    return retrieved_sections, retrieval_passed

def evaluate_case(
    evaluation_case: dict[str, str],
    index: dict[str, Any],
    client: Any,
    embedding_model: str,
    llm_model: str,
    max_output_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    """Run and evaluate one end-to-end RAG test case."""
    rag_output = _run_evaluation_query(
        evaluation_case, index, client,
        embedding_model, llm_model,
        max_output_tokens, top_k,
    )
    retrieved_sections, retrieval_passed = _evaluate_retrieval(
        evaluation_case,
        rag_output["chunks_related"],
    )
    answer_evaluation = evaluate_rag_output(
        rag_output=rag_output,
        client=client,
        model=llm_model,
    )

    return _build_case_result(
        evaluation_case, rag_output,
        retrieved_sections, retrieval_passed,
        answer_evaluation,
    )

def _count_passed(
    results: list[dict[str, Any]],
    field: str,
) -> int:
    """Count results whose selected boolean field is true."""
    return sum(
        bool(result[field])
        for result in results
    )

def build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate retrieval, answer, and overall metrics."""
    total = len(results)
    retrieval_passed = _count_passed(
        results,
        "retrieval_passed",
    )
    answer_passed = _count_passed(
        results,
        "answer_passed",
    )
    fully_passed = _count_passed(
        results,
        "passed",
    )

    return {
        "total_cases": total,
        "retrieval_passed": retrieval_passed,
        "retrieval_accuracy": round(retrieval_passed / total, 4),
        "answer_passed": answer_passed,
        "answer_accuracy": round(answer_passed / total, 4),
        "fully_passed": fully_passed,
        "overall_accuracy": round(fully_passed / total, 4),
    }

def _select_evaluation_cases(
    dataset: list[dict[str, str]],
    limit: int | None,
) -> list[dict[str, str]]:
    """Apply an optional positive limit to evaluation cases."""
    if limit is None:
        return dataset

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero."
        )

    return dataset[:limit]


def _evaluate_cases(
    dataset: list[dict[str, str]],
    index: dict[str, Any],
    client: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    """Execute every selected evaluation case."""
    results = []

    for position, evaluation_case in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"Evaluating {position}/{len(dataset)}: "
            f"{evaluation_case['id']}"
        )
        results.append(
            evaluate_case(
                evaluation_case=evaluation_case,
                index=index,
                client=client,
                embedding_model=get_embedding_model(),
                llm_model=get_llm_model(),
                max_output_tokens=get_max_output_tokens(),
                top_k=top_k,
            )
        )

    return results


def _save_report(
    report: dict[str, Any],
    report_path: Path,
) -> None:
    """Persist an evaluation report as formatted JSON."""
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

def _load_evaluation_index(
    index_path: Path,
) -> dict[str, Any]:
    """Load an index compatible with the configured embedding model."""
    index = load_index(index_path)

    validate_index_model(
        index,
        get_embedding_model(),
    )

    return index

def run_evaluation(
    dataset_path: Path,
    index_path: Path,
    report_path: Path,
    top_k: int,
    limit: int | None,
) -> dict[str, Any]:
    """Evaluate the dataset and persist the JSON report."""
    dataset = _select_evaluation_cases(
        load_evaluation_dataset(dataset_path),
        limit,
    )
    results = _evaluate_cases(
        dataset=dataset,
        index=_load_evaluation_index(index_path),
        client=get_openai_client(),
        top_k=top_k,
    )
    report = {
        "summary": build_summary(results),
        "results": results,
    }

    _save_report(report, report_path)

    return report

def _build_evaluation_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for RAG evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the AR HR FAQ RAG system."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases.",
    )

    return parser

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation command."""
    return _build_evaluation_parser().parse_args()


def main() -> None:
    """Run the configured evaluation and print its summary."""
    args = parse_arguments()

    report = run_evaluation(
        dataset_path=args.dataset,
        index_path=args.index,
        report_path=args.report,
        top_k=args.top_k,
        limit=args.limit,
    )

    print("\nEvaluation completed")
    print(
        json.dumps(
            report["summary"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()