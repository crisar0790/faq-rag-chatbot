"""Provide an interactive command-line interface for the FAQ chatbot."""

import json
from collections.abc import Callable
from typing import Any

from src.query import create_query_runner
from src.errors import format_error

EXIT_COMMANDS = {"exit", "quit"}

WELCOME_MESSAGE = (
    "AR HR FAQ Chatbot\n"
    "Ask a question about the available documentation.\n"
    "Type 'exit' or 'quit' to close the chatbot."
)

EMPTY_QUESTION_MESSAGE = "Please enter a question."
GOODBYE_MESSAGE = "Goodbye!"

def format_result(result: dict[str, Any]) -> str:
    """Serialize a RAG result as readable JSON."""
    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

def _read_question(input_function: Callable[[str], str]) -> str | None:
    """Read one question or return None when input closes."""
    try:
        return input_function("\nQuestion: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

def run_chat(
    ask_question: Callable[[str], dict[str, Any]],
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    """Run the interactive question-and-answer loop until the user exits."""
    output_function(WELCOME_MESSAGE)

    while True:
        question = _read_question(input_function)

        if question is None:
            output_function(GOODBYE_MESSAGE)
            break

        if question.lower() in EXIT_COMMANDS:
            output_function(GOODBYE_MESSAGE)
            break

        if not question:
            output_function(EMPTY_QUESTION_MESSAGE)
            continue

        try:
            result = ask_question(question)
        except Exception as error:
            output_function(format_error(error))
            continue

        output_function(format_result(result))

def main() -> None:
    """Start the interactive chatbot command."""
    try:
        query_runner = create_query_runner()
    except Exception as error:
        print(format_error(error))
        raise SystemExit(1) from error

    run_chat(ask_question=query_runner)

if __name__ == "__main__":
    main()