import json
from pathlib import Path

from cola_dlm.evaluate_cli import main


def test_evaluate_cli_scores_free_form_and_multiple_choice_jsonl(tmp_path):
    input_path = tmp_path / "examples.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "free",
                "task": "lambada",
                "context": "The explorer found a hidden",
                "answer": "map",
                "generation": "Completion: map",
            },
            {
                "id": "mc",
                "task": "mmlu",
                "question": "Which organ pumps blood?",
                "choices": ["heart", "lung", "skin", "bone"],
                "answer": "heart",
                "generation": "A. heart",
            },
        ],
    )

    assert main(
        [
            "--input",
            str(input_path),
            "--output",
            str(metrics_path),
            "--summary-output",
            str(summary_path),
        ]
    ) == 0

    metrics = _read_jsonl(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [metric["id"] for metric in metrics] == ["free", "mc"]
    assert all(metric["exact_match"] for metric in metrics)
    assert metrics[0]["task"] == "lambada"
    assert metrics[1]["task"] == "mmlu"
    assert summary["num_examples"] == 2
    assert summary["exact_matches"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["tasks"]["lambada"]["accuracy"] == 1.0
    assert summary["tasks"]["mmlu"]["accuracy"] == 1.0


def test_evaluate_cli_uses_default_task_for_jsonl_records(tmp_path):
    input_path = tmp_path / "examples.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "qa",
                "context": "Ada wrote the notes in London.",
                "question": "Where did Ada write the notes?",
                "answer": "London",
                "generation": "Answer: London",
            },
        ],
    )

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(metrics_path),
            "--task",
            "squad",
        ]
    )

    metrics = _read_jsonl(metrics_path)
    summary = json.loads(
        metrics_path.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert metrics[0]["task"] == "squad"
    assert metrics[0]["exact_match"] is True
    assert metrics[0]["f1"] == 1.0
    assert summary["average_f1"] == 1.0


def test_evaluate_console_script_is_registered():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'evaluate = "cola_dlm.evaluate_cli:main"' in pyproject


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
