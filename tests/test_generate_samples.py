import json

from src.generate_samples import (
    generate_samples,
    save_samples,
)

from unittest.mock import Mock

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

def test_main_reuses_one_query_runner(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "sample_queries.json"
    query_runner = Mock(
        side_effect=lambda question: {
            "user_question": question,
            "system_answer": "Test answer.",
            "chunks_related": [
                {
                    "chunk_id": "chunk_001",
                    "section": "Test Section",
                    "text": "First source.",
                },
                {
                    "chunk_id": "chunk_002",
                    "section": "Test Section",
                    "text": "Second source.",
                },
            ],
        }
    )
    create_runner_mock = Mock(
        return_value=query_runner
    )

    monkeypatch.setattr(
        "src.generate_samples.create_query_runner",
        create_runner_mock,
    )
    monkeypatch.setattr(
        "src.generate_samples.OUTPUT_PATH",
        output_path,
    )

    from src.generate_samples import main

    main()

    create_runner_mock.assert_called_once()
    assert query_runner.call_count == 3
    assert output_path.exists()