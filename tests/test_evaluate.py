from src.evaluate import (
    build_evaluation_input,
    build_summary,
)


def test_build_evaluation_input():
    result = build_evaluation_input(
        question="How can I reset my password?",
        answer="Use the password reset option.",
        chunks=[
            {
                "chunk_id": "chunk_001",
                "section": "Account Access",
                "text": "Users can reset their password.",
            }
        ],
    )

    assert "How can I reset my password?" in result
    assert "Use the password reset option." in result
    assert "chunk_001" in result
    assert "Users can reset their password." in result


def test_build_summary():
    results = [
        {
            "retrieval_passed": True,
            "answer_passed": True,
            "passed": True,
        },
        {
            "retrieval_passed": True,
            "answer_passed": False,
            "passed": False,
        },
        {
            "retrieval_passed": False,
            "answer_passed": True,
            "passed": False,
        },
    ]

    summary = build_summary(results)

    assert summary == {
        "total_cases": 3,
        "retrieval_passed": 2,
        "retrieval_accuracy": 0.6667,
        "answer_passed": 2,
        "answer_accuracy": 0.6667,
        "fully_passed": 1,
        "overall_accuracy": 0.3333,
    }