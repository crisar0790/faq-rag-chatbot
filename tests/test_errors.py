import json

from src.answer_generator import AnswerGenerationError
from src.errors import format_error, get_error_message


def test_formats_file_not_found_error():
    error = FileNotFoundError("data/index.json")

    message = get_error_message(error)

    assert "required file was not found" in message


def test_formats_answer_generation_error():
    error = AnswerGenerationError(
        "The model returned invalid JSON."
    )

    message = get_error_message(error)

    assert "answer could not be generated" in message
    assert "invalid JSON" in message


def test_formats_value_error():
    error = ValueError(
        "top_k must be between 2 and 5."
    )

    message = get_error_message(error)

    assert message == "top_k must be between 2 and 5."


def test_hides_unexpected_error_details():
    error = RuntimeError(
        "Sensitive internal implementation details"
    )

    message = get_error_message(error)

    assert message == "An unexpected error occurred."
    assert "Sensitive" not in message


def test_format_error_returns_json():
    formatted_error = format_error(
        ValueError("Question cannot be empty.")
    )

    parsed_error = json.loads(formatted_error)

    assert parsed_error == {
        "error": "Question cannot be empty."
    }