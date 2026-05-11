"""Command line smoke evaluator for local JSONL examples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cola_dlm.evaluation import (
    PromptResult,
    build_hellaswag_prompt,
    build_lambada_prompt,
    build_mmlu_prompt,
    build_obqa_prompt,
    build_race_prompt,
    build_siqa_prompt,
    build_squad_prompt,
    build_story_cloze_prompt,
    score_lambada_answer,
    score_multiple_choice_answer,
    score_squad_answer,
)


FREE_FORM_TASKS = {"lambada", "squad"}
MULTIPLE_CHOICE_TASKS = {
    "mmlu",
    "siqa",
    "story_cloze",
    "obqa",
    "race",
    "hellaswag",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Score local JSONL examples from command line arguments."""

    args = _build_parser().parse_args(argv)
    evaluate_file(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        default_task=args.task,
    )
    return 0


def evaluate_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    default_task: str | None = None,
) -> dict[str, Any]:
    """Score examples and write JSONL metrics plus an aggregate JSON summary."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if summary_path is None:
        summary_path = output_path.with_suffix(".summary.json")
    summary_path = Path(summary_path)

    metrics = [
        _score_record(record, default_task=default_task, line_number=line_number)
        for line_number, record in _iter_jsonl_records(input_path)
    ]
    summary = _aggregate(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics),
        encoding="utf-8",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score local JSONL generations with Cola DLM prompt helpers.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL examples.")
    parser.add_argument("--output", required=True, help="Per-example metrics JSONL.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Aggregate summary JSON. Defaults next to --output.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Default task when JSONL records do not include a task field.",
    )
    return parser


def _iter_jsonl_records(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    records: list[tuple[int, Mapping[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Evaluation input not found: {path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")
        records.append((line_number, record))
    return records


def _score_record(
    record: Mapping[str, Any],
    *,
    default_task: str | None,
    line_number: int,
) -> dict[str, Any]:
    try:
        task = _resolve_task(record, default_task)
        generation = _required_text(record, "generation")
        prompt = _build_prompt(task, record)
        metric = _score_generation(task, generation, prompt)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {exc}") from exc

    return {
        "id": record.get("id", line_number),
        "task": task,
        "prompt_chars": len(prompt.prompt),
        "max_new_tokens": prompt.max_new_tokens,
        **metric,
    }


def _resolve_task(record: Mapping[str, Any], default_task: str | None) -> str:
    raw_task = record.get("task", default_task)
    if not isinstance(raw_task, str) or not raw_task.strip():
        raise ValueError("record must include task or --task must be provided")
    task = raw_task.strip().lower()
    if task not in FREE_FORM_TASKS and task not in MULTIPLE_CHOICE_TASKS:
        supported = sorted(FREE_FORM_TASKS | MULTIPLE_CHOICE_TASKS)
        raise ValueError(f"unsupported task {raw_task!r}; expected one of {supported}")
    return task


def _build_prompt(task: str, record: Mapping[str, Any]) -> PromptResult:
    examples = record.get("examples", ())
    max_new_tokens = _max_new_tokens(record)
    if task == "lambada":
        return build_lambada_prompt(
            _required_text(record, "context"),
            _required_text(record, "answer"),
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
    if task == "squad":
        return build_squad_prompt(
            context=_required_text(record, "context"),
            question=_required_text(record, "question"),
            answer=_required_text(record, "answer"),
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
    if task == "mmlu":
        return build_mmlu_prompt(
            question=_required_text(record, "question"),
            choices=_required_choices(record),
            answer=_required_text(record, "answer"),
            examples=examples,
            subject=_optional_text(record, "subject"),
            max_new_tokens=max_new_tokens,
        )
    if task == "siqa":
        return build_siqa_prompt(
            context=_required_text(record, "context"),
            question=_required_text(record, "question"),
            choices=_required_choices(record),
            answer=_required_text(record, "answer"),
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
    if task == "story_cloze":
        return build_story_cloze_prompt(
            story=_required_text(record, "story"),
            choices=_required_choices(record),
            answer=_required_text(record, "answer"),
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
    if task == "obqa":
        return build_obqa_prompt(
            question=_required_text(record, "question"),
            choices=_required_choices(record),
            answer=_required_text(record, "answer"),
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
    if task == "race":
        return build_race_prompt(
            passage=_required_text(record, "passage"),
            question=_required_text(record, "question"),
            choices=_required_choices(record),
            answer=_required_text(record, "answer"),
            examples=examples,
            max_new_tokens=max_new_tokens,
        )
    return build_hellaswag_prompt(
        context=_required_text(record, "context"),
        choices=_required_choices(record),
        answer=_required_text(record, "answer"),
        examples=examples,
        activity_label=_optional_text(record, "activity_label"),
        max_new_tokens=max_new_tokens,
    )


def _score_generation(
    task: str,
    generation: str,
    prompt: PromptResult,
) -> dict[str, Any]:
    if task == "lambada":
        score = score_lambada_answer(generation, prompt.answer)
        return {
            "exact_match": score.exact_match,
            "normalized_generation": score.normalized_generation,
            "normalized_target": score.normalized_target,
        }
    if task == "squad":
        score = score_squad_answer(generation, prompt.answer)
        return {
            "exact_match": score.exact_match,
            "f1": score.f1,
            "normalized_generation": score.normalized_generation,
            "normalized_answers": list(score.normalized_answers),
        }

    score = score_multiple_choice_answer(generation, prompt.answer, prompt.choices)
    return {
        "exact_match": score.exact_match,
        "normalized_generation": score.normalized_generation,
        "normalized_answer": score.normalized_answer,
    }


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_groups: dict[str, dict[str, Any]] = {}
    exact_matches = 0
    f1_total = 0.0
    f1_count = 0

    for metric in metrics:
        task = str(metric["task"])
        group = task_groups.setdefault(
            task,
            {"num_examples": 0, "exact_matches": 0, "f1_total": 0.0, "f1_count": 0},
        )
        group["num_examples"] += 1
        if metric["exact_match"]:
            exact_matches += 1
            group["exact_matches"] += 1
        if "f1" in metric:
            f1_total += float(metric["f1"])
            f1_count += 1
            group["f1_total"] += float(metric["f1"])
            group["f1_count"] += 1

    tasks = {
        task: _finalize_group(group)
        for task, group in sorted(task_groups.items())
    }
    summary: dict[str, Any] = {
        "num_examples": len(metrics),
        "exact_matches": exact_matches,
        "accuracy": _safe_divide(exact_matches, len(metrics)),
        "tasks": tasks,
    }
    if f1_count:
        summary["average_f1"] = f1_total / f1_count
    return summary


def _finalize_group(group: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "num_examples": group["num_examples"],
        "exact_matches": group["exact_matches"],
        "accuracy": _safe_divide(group["exact_matches"], group["num_examples"]),
    }
    if group["f1_count"]:
        result["average_f1"] = group["f1_total"] / group["f1_count"]
    return result


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _max_new_tokens(record: Mapping[str, Any]) -> int:
    value = record.get("max_new_tokens", 32)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return value


def _required_text(record: Mapping[str, Any], field: str) -> str:
    if field not in record:
        raise ValueError(f"record is missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _optional_text(record: Mapping[str, Any], field: str) -> str | None:
    if field not in record:
        return None
    return _required_text(record, field)


def _required_choices(record: Mapping[str, Any]) -> Sequence[str]:
    if "choices" not in record:
        raise ValueError("record is missing required field 'choices'")
    choices = record["choices"]
    if isinstance(choices, str) or not isinstance(choices, Sequence):
        raise ValueError("choices must be an ordered sequence of strings")
    return choices


if __name__ == "__main__":
    raise SystemExit(main())
