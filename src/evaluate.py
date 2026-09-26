import argparse
import json
from pathlib import Path
from typing import Any

from src.config import (
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
from src.vector_store import load_index


DEFAULT_REPORT_PATH = Path("evaluation/report.json")
EVALUATOR_PROMPT_PATH = Path(
    "prompts/evaluator_prompt.md"
)

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {
            "type": "boolean",
        },
        "relevant": {
            "type": "boolean",
        },
        "complete": {
            "type": "boolean",
        },
        "explanation": {
            "type": "string",
        },
    },
    "required": [
        "grounded",
        "relevant",
        "complete",
        "explanation",
    ],
    "additionalProperties": False,
}


def load_evaluator_prompt(path: Path = EVALUATOR_PROMPT_PATH) -> str:
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


def evaluate_answer(question: str, answer: str, chunks: list[dict[str, Any]], client: Any, model: str) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": load_evaluator_prompt(),
            },
            {
                "role": "user",
                "content": build_evaluation_input(
                    question=question,
                    answer=answer,
                    chunks=chunks,
                ),
            },
        ],
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

    if response.status != "completed":
        raise ValueError(
            "The evaluator response was not completed."
        )

    try:
        evaluation = json.loads(
            response.output_text
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "The evaluator returned invalid JSON."
        ) from error

    return evaluation


def evaluate_case(evaluation_case: dict[str, str], index: dict[str, Any], client: Any, embedding_model: str, llm_model: str, max_output_tokens: int, top_k: int) -> dict[str, Any]:
    rag_output = answer_question(
        question=evaluation_case["question"],
        index=index,
        client=client,
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_output_tokens=max_output_tokens,
        create_query_embedding=(
            create_query_embedding
        ),
        retrieve_chunks=retrieve_chunks,
        top_k=top_k,
    )

    retrieved_sections = list(
        dict.fromkeys(
            chunk["section"]
            for chunk in rag_output["chunks_related"]
        )
    )

    retrieval_passed = (
        evaluation_case["expected_section"]
        in retrieved_sections
    )

    answer_evaluation = evaluate_answer(
        question=evaluation_case["question"],
        answer=rag_output["system_answer"],
        chunks=rag_output["chunks_related"],
        client=client,
        model=llm_model,
    )

    answer_passed = all(
        [
            answer_evaluation["grounded"],
            answer_evaluation["relevant"],
            answer_evaluation["complete"],
        ]
    )

    return {
        "id": evaluation_case["id"],
        "question": evaluation_case["question"],
        "expected_section": (
            evaluation_case["expected_section"]
        ),
        "retrieved_sections": retrieved_sections,
        "retrieval_passed": retrieval_passed,
        "system_answer": rag_output["system_answer"],
        "answer_evaluation": answer_evaluation,
        "answer_passed": answer_passed,
        "passed": retrieval_passed and answer_passed,
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)

    retrieval_passed = sum(
        result["retrieval_passed"]
        for result in results
    )
    answer_passed = sum(
        result["answer_passed"]
        for result in results
    )
    fully_passed = sum(
        result["passed"]
        for result in results
    )

    return {
        "total_cases": total,
        "retrieval_passed": retrieval_passed,
        "retrieval_accuracy": round(
            retrieval_passed / total,
            4,
        ),
        "answer_passed": answer_passed,
        "answer_accuracy": round(
            answer_passed / total,
            4,
        ),
        "fully_passed": fully_passed,
        "overall_accuracy": round(
            fully_passed / total,
            4,
        ),
    }


def run_evaluation(dataset_path: Path, index_path: Path, report_path: Path, top_k: int, limit: int | None) -> dict[str, Any]:
    dataset = load_evaluation_dataset(
        dataset_path
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        dataset = dataset[:limit]

    index = load_index(index_path)
    client = get_openai_client()

    results = []

    for position, evaluation_case in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"Evaluating {position}/{len(dataset)}: "
            f"{evaluation_case['id']}"
        )

        result = evaluate_case(
            evaluation_case=evaluation_case,
            index=index,
            client=client,
            embedding_model=get_embedding_model(),
            llm_model=get_llm_model(),
            max_output_tokens=get_max_output_tokens(),
            top_k=top_k,
        )

        results.append(result)

    report = {
        "summary": build_summary(results),
        "results": results,
    }

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

    return report


def parse_arguments() -> argparse.Namespace:
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases.",
    )

    return parser.parse_args()


def main() -> None:
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