import json

from src.chat import (
    EMPTY_QUESTION_MESSAGE,
    GOODBYE_MESSAGE,
    WELCOME_MESSAGE,
    format_result,
    run_chat,
)


def test_format_result_returns_formatted_json():
    result = {
        "user_question": "How can I reset my password?",
        "system_answer": "Use the password reset option.",
        "chunks_related": [],
    }

    formatted_result = format_result(result)
    parsed_result = json.loads(formatted_result)

    assert parsed_result == result
    assert "\n" in formatted_result


def test_run_chat_answers_question_and_exits():
    answers = iter(
        [
            "How can I reset my password?",
            "exit",
        ]
    )
    outputs = []
    received_questions = []

    def fake_input(prompt):
        return next(answers)

    def fake_output(message):
        outputs.append(message)

    def fake_ask_question(question):
        received_questions.append(question)

        return {
            "user_question": question,
            "system_answer": (
                "Use the password reset option."
            ),
            "chunks_related": [
                {
                    "chunk_id": "chunk_001",
                    "section": "Account Access",
                    "text": "Password reset information.",
                }
            ],
        }

    run_chat(
        ask_question=fake_ask_question,
        input_function=fake_input,
        output_function=fake_output,
    )

    assert received_questions == [
        "How can I reset my password?"
    ]
    assert outputs[0] == WELCOME_MESSAGE
    assert "system_answer" in outputs[1]
    assert outputs[2] == GOODBYE_MESSAGE


def test_run_chat_rejects_empty_question():
    answers = iter(
        [
            "   ",
            "quit",
        ]
    )
    outputs = []
    received_questions = []

    def fake_input(prompt):
        return next(answers)

    def fake_output(message):
        outputs.append(message)

    def fake_ask_question(question):
        received_questions.append(question)
        return {}

    run_chat(
        ask_question=fake_ask_question,
        input_function=fake_input,
        output_function=fake_output,
    )

    assert received_questions == []
    assert EMPTY_QUESTION_MESSAGE in outputs
    assert outputs[-1] == GOODBYE_MESSAGE


def test_exit_command_is_case_insensitive():
    answers = iter(["EXIT"])
    outputs = []

    run_chat(
        ask_question=lambda question: {},
        input_function=lambda prompt: next(answers),
        output_function=outputs.append,
    )

    assert outputs[-1] == GOODBYE_MESSAGE

def test_chat_continues_after_query_error():
    answers = iter(
        [
            "First question",
            "Second question",
            "exit",
        ]
    )
    outputs = []
    call_count = 0

    def fake_ask_question(question):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise ValueError("Temporary query error.")

        return {
            "user_question": question,
            "system_answer": "Successful answer.",
            "chunks_related": [],
        }

    run_chat(
        ask_question=fake_ask_question,
        input_function=lambda prompt: next(answers),
        output_function=outputs.append,
    )

    assert call_count == 2
    assert '"error": "Temporary query error."' in outputs[1]
    assert '"system_answer": "Successful answer."' in outputs[2]
    assert outputs[-1] == GOODBYE_MESSAGE