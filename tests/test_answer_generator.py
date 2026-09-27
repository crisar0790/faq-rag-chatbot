import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.answer_generator import (
    AnswerGenerationError,
    _parse_answer_response,
    build_context,
    build_user_message,
    generate_grounded_answer,
    load_answer_prompt,
)

class FakeResponsesAPI:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponsesAPI(response)


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": "chunk_001",
            "section": "Account Access",
            "text": "Employees can reset their password.",
        },
        {
            "chunk_id": "chunk_002",
            "section": "Account Access",
            "text": "Password reset links expire after 30 minutes.",
        },
    ]


def test_load_answer_prompt(tmp_path: Path):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(
        "Answer using only the context.",
        encoding="utf-8",
    )

    prompt = load_answer_prompt(prompt_path)

    assert prompt == "Answer using only the context."


def test_build_context(chunks):
    context = build_context(chunks)

    assert "chunk_001" in context
    assert "Account Access" in context
    assert "Employees can reset their password." in context


def test_build_user_message(chunks):
    message = build_user_message(
        "How can I reset my password?",
        chunks,
    )

    assert "CONTEXT" in message
    assert "USER QUESTION" in message
    assert "How can I reset my password?" in message


def test_generate_grounded_answer(chunks):
    response = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {
                "system_answer": (
                    "Employees can reset their password, and "
                    "the reset link expires after 30 minutes."
                )
            }
        ),
    )
    client = FakeClient(response)

    answer = generate_grounded_answer(
        question="How can I reset my password?",
        chunks=chunks,
        client=client,
        model="gpt-4o-mini",
        max_output_tokens=400,
    )

    assert "30 minutes" in answer

    request = client.responses.last_request
    assert request["model"] == "gpt-4o-mini"
    assert request["text"]["format"]["strict"] is True
    assert (
        request["text"]["format"]["schema"]
        ["additionalProperties"]
        is False
    )


def test_rejects_incomplete_response(chunks):
    response = SimpleNamespace(
        status="incomplete",
        output_text="",
        incomplete_details=SimpleNamespace(
            reason="max_output_tokens"
        ),
    )
    client = FakeClient(response)

    with pytest.raises(
        AnswerGenerationError,
        match="max_output_tokens",
    ):
        generate_grounded_answer(
            question="How can I reset my password?",
            chunks=chunks,
            client=client,
            model="gpt-4o-mini",
            max_output_tokens=10,
        )


def test_rejects_invalid_json(chunks):
    response = SimpleNamespace(
        status="completed",
        output_text="not valid JSON",
    )
    client = FakeClient(response)

    with pytest.raises(
        AnswerGenerationError,
        match="invalid JSON",
    ):
        generate_grounded_answer(
            question="How can I reset my password?",
            chunks=chunks,
            client=client,
            model="gpt-4o-mini",
            max_output_tokens=400,
        )


def test_rejects_empty_answer(chunks):
    response = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {"system_answer": "   "}
        ),
    )
    client = FakeClient(response)

    with pytest.raises(
        AnswerGenerationError,
        match="empty answer",
    ):
        generate_grounded_answer(
            question="How can I reset my password?",
            chunks=chunks,
            client=client,
            model="gpt-4o-mini",
            max_output_tokens=400,
        )

def test_parse_answer_response():
    response = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {
                "system_answer": (
                    "Employees can reset their password "
                    "from the login page."
                )
            }
        ),
    )

    answer = _parse_answer_response(response)

    assert answer == (
        "Employees can reset their password "
        "from the login page."
    )


def test_parse_answer_response_rejects_incomplete_result():
    response = SimpleNamespace(
        status="incomplete",
        output_text="",
        incomplete_details=SimpleNamespace(
            reason="max_output_tokens"
        ),
    )

    with pytest.raises(
        AnswerGenerationError,
        match="max_output_tokens",
    ):
        _parse_answer_response(response)