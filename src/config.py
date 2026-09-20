"""Base project configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

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
