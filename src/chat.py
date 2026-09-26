import json
from collections.abc import Callable
from typing import Any

from src.query import run_query

EXIT_COMMANDS = {"exit", "quit"}

WELCOME_MESSAGE = (
    "AR HR FAQ Chatbot\n"
    "Ask a question about the available documentarion.\n"
    "Type 'exit' or 'quit' to close the chatbot."
)

EMPTY_QUESTION_MESSAGE = "Please enter a question."
GOODBYE_MESSAGE = "Goodbye!"

def format_result(result: dict[str, Any]) -> str:
    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

def run_chat(
    ask_question: Callable[[str], dict[str, Any]] = run_query,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    output_function(WELCOME_MESSAGE)

    while True:
        question = input_function("\nQuestion: ").strip()

        if question.lower() in EXIT_COMMANDS:
            output_function(GOODBYE_MESSAGE)
            break

        if not question:
            output_function(EMPTY_QUESTION_MESSAGE)
            continue

        result = ask_question(question)
        output_function(format_result(result))

def main() -> None:
    run_chat()

if __name__ == "__main__":
    main()