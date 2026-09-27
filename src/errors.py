"""Convert internal exceptions into safe user-facing error messages."""

import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)

from src.answer_generator import AnswerGenerationError

def get_error_message(error: Exception) -> str:
    """Return a safe and understandable message for an application error."""
    if isinstance(error, FileNotFoundError):
        return (
            "The required file was not found. "
            "Verify that the vector index and prompt exist."
        )

    if isinstance(error, AuthenticationError):
        return (
            "OpenAI authentication failed. "
            "Verify the OPENAI_API_KEY value."
        )

    if isinstance(error, APITimeoutError):
        return (
            "The OpenAI request timed out. "
            "Please try again."
        )

    if isinstance(error, APIConnectionError):
        return (
            "Could not connect to OpenAI. "
            "Check your internet connection and try again."
        )

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
    """Serialize an application error using the public JSON error format."""
    return json.dumps(
        {
            "error": get_error_message(error),
        },
        indent=2,
        ensure_ascii=False,
    )
