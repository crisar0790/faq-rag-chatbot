import json

from src.generate_samples import (
    generate_samples,
    save_samples,
)


def test_generate_samples_runs_every_question():
    received_questions = []

    def fake_query(question):
        received_questions.append(question)

        return {
            "user_question": question,
            "system_answer": "Test answer.",
            "chunks_related": [
                {
                    "chunk_id": "chunk_001",
                    "section": "Test Section",
                    "text": "Test source text.",
                },
                {
                    "chunk_id": "chunk_002",
                    "section": "Test Section",
                    "text": "Additional source text.",
                },
            ],
        }

    samples = generate_samples(
        questions=["First question", "Second question"],
        query_function=fake_query,
    )

    assert received_questions == [
        "First question",
        "Second question",
    ]
    assert len(samples) == 2


def test_save_samples_creates_json_file(tmp_path):
    output_path = tmp_path / "sample_queries.json"

    samples = [
        {
            "user_question": "Test question",
            "system_answer": "Test answer.",
            "chunks_related": [],
        }
    ]

    save_samples(
        samples=samples,
        output_path=output_path,
    )

    saved_samples = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_samples == samples