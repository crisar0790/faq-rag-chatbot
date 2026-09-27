"""Generate grounded answers from retrieved document chunks."""

import json
from pathlib import Path
from typing import Any


PROMPT_PATH = Path("prompts/answer_prompt.md")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "system_answer": {
            "type": "string",
        }
    },
    "required": ["system_answer"],
    "additionalProperties": False,
}


class AnswerGenerationError(Exception):
    """Raised when a grounded answer cannot be generated."""


def load_answer_prompt(prompt_path: Path = PROMPT_PATH) -> str:
    """Load and validate the system prompt used for answer generation."""
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Answer prompt not found: {prompt_path}"
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError("Answer prompt cannot be empty.")

    return prompt


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as contextual evidence for the model."""
    if not chunks:
        raise ValueError("At least one chunk is required.")

    context_parts = []

    for chunk in chunks:
        context_parts.append(
            "\n".join(
                [
                    (
                        f"[{chunk['chunk_id']} | "
                        f"{chunk['section']}]"
                    ),
                    chunk["text"],
                ]
            )
        )

    return "\n\n".join(context_parts)


def build_user_message(question: str, chunks: list[dict[str, Any]]) -> str:
    """Combine the user question and retrieved context into one message."""
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    context = build_context(chunks)

    return (
        "CONTEXT\n"
        "-------\n"
        f"{context}\n\n"
        "USER QUESTION\n"
        "-------------\n"
        f"{clean_question}"
    )


def generate_grounded_answer(question: str, chunks: list[dict[str, Any]], client: Any, model: str, max_output_tokens: int) -> str:
    """Generate and validate an answer grounded in the retrieved chunks."""
    prompt = load_answer_prompt()
    user_message = build_user_message(question, chunks)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            }
        },
    )

    if response.status != "completed":
        reason = getattr(
            getattr(response, "incomplete_details", None),
            "reason",
            "unknown",
        )
        raise AnswerGenerationError(
            f"Answer generation was not completed: {reason}"
        )

    try:
        result = json.loads(response.output_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise AnswerGenerationError(
            "The model returned invalid JSON."
        ) from error

    answer = result.get("system_answer")

    if not isinstance(answer, str) or not answer.strip():
        raise AnswerGenerationError(
            "The model returned an empty answer."
        )

    return answer.strip()