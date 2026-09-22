"""Base project configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
DOCUMENT_PATH = DATA_DIR / "faq_document.txt"
INDEX_PATH = DATA_DIR / "index.json"

load_dotenv(ENV_PATH)

def get_embedding_model() -> str:
    """Return the configured OpenAI embedding model."""
    return os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small",
    ).strip()

def get_openai_timeout() -> float:
    """Return the configured positive API timeout."""
    raw_timeout = os.getenv("OPENAI_TIMEOUT", "30").strip()

    try:
        timeout = float(raw_timeout)
    except ValueError as error:
        raise ValueError(
            "OPENAI_TIMEOUT must be a number."
        ) from error

    if timeout <= 0:
        raise ValueError(
            "OPENAI_TIMEOUT must be greater than zero."
        )

    return timeout


def get_openai_client() -> OpenAI:
    """Create an authenticated OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Create .env from .env.example."
        )

    return OpenAI(
        api_key=api_key,
        timeout=get_openai_timeout(),
    )