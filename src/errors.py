"""Convert internal exceptions into safe user-facing error messages."""

import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)

from src.answer_generator import AnswerGenerationError


STATIC_ERROR_MESSAGES = (
    (
        FileNotFoundError,
        (
            "The required file was not found. "
            "Verify that the vector index and prompt exist."
        ),
    ),
    (
        AuthenticationError,
        (
            "OpenAI authentication failed. "
            "Verify the OPENAI_API_KEY value."
        ),
    ),
    (
        APITimeoutError,
        (
            "The OpenAI request timed out. "
            "Please try again."
        ),
    ),
    (
        APIConnectionError,
        (
            "Could not connect to OpenAI. "
            "Check your internet connection and try again."
        ),
    ),
)


def _get_static_error_message(
    error: Exception,
) -> str | None:
    """Return a predefined message for a known exception."""
    for error_type, message in STATIC_ERROR_MESSAGES:
        if isinstance(error, error_type):
            return message

    return None


def get_error_message(error: Exception) -> str:
    """Return a safe and understandable application error."""
    static_message = _get_static_error_message(error)

    if static_message is not None:
        return static_message

    if isinstance(error, APIStatusError):
        return (
            "OpenAI returned an API error "
            f"with status code {error.status_code}."
        )

    if isinstance(error, AnswerGenerationError):
        return (
            "The answer could not be generated. "
            f"{error}"
        )

    if isinstance(error, ValueError):
        return str(error)

    return "An unexpected error occurred."


def format_error(error: Exception) -> str:
    """Serialize an error using the public JSON error format."""
    return json.dumps(
        {"error": get_error_message(error)},
        indent=2,
        ensure_ascii=False,
    )