"""Prompt builders for small benchmark evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cola_dlm.config import InferenceConfig


DEFAULT_MAX_NEW_TOKENS = InferenceConfig().max_new_tokens
PromptExample = Mapping[str, Any]


@dataclass(frozen=True)
class PromptResult:
    """Formatted prompt plus the canonical answer metadata."""

    prompt: str
    answer: str
    choices: tuple[str, ...] = ()
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


def build_lambada_prompt(
    context: str,
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a LAMBADA final-word completion prompt."""

    blocks = [
        _format_lambada_block(
            context=_required_text(example, "context"),
            answer=_required_text(example, "answer"),
        )
        for example in examples
    ]
    blocks.append(_format_lambada_block(context=_clean_text(context, "context")))
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=_clean_text(answer, "answer"),
        max_new_tokens=max_new_tokens,
    )


def build_mmlu_prompt(
    question: str,
    choices: Sequence[str],
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    subject: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a multiple-choice MMLU prompt."""

    choice_texts = _clean_choices(choices)
    answer_text = _clean_answer_choice(answer, choice_texts)
    blocks = [_format_mmlu_example(example) for example in examples]
    blocks.append(
        _format_mmlu_block(
            question=_clean_text(question, "question"),
            choices=choice_texts,
            subject=_optional_clean_text(subject, "subject"),
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=answer_text,
        choices=choice_texts,
        max_new_tokens=max_new_tokens,
    )


def build_siqa_prompt(
    context: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a Social IQA multiple-choice prompt."""

    choice_texts = _clean_choices(choices)
    answer_text = _clean_answer_choice(answer, choice_texts)
    blocks = [_format_siqa_example(example) for example in examples]
    blocks.append(
        _format_siqa_block(
            context=_clean_text(context, "context"),
            question=_clean_text(question, "question"),
            choices=choice_texts,
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=answer_text,
        choices=choice_texts,
        max_new_tokens=max_new_tokens,
    )


def build_squad_prompt(
    context: str,
    question: str,
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a SQuAD-style extractive question-answering prompt."""

    blocks = [_format_squad_example(example) for example in examples]
    blocks.append(
        _format_squad_block(
            context=_clean_text(context, "context"),
            question=_clean_text(question, "question"),
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=_clean_text(answer, "answer"),
        max_new_tokens=max_new_tokens,
    )


def build_story_cloze_prompt(
    story: str,
    choices: Sequence[str],
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a Story Cloze ending-selection prompt."""

    choice_texts = _clean_choices(choices)
    answer_text = _clean_answer_choice(answer, choice_texts)
    blocks = [_format_story_cloze_example(example) for example in examples]
    blocks.append(
        _format_story_cloze_block(
            story=_clean_text(story, "story"),
            choices=choice_texts,
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=answer_text,
        choices=choice_texts,
        max_new_tokens=max_new_tokens,
    )


def build_obqa_prompt(
    question: str,
    choices: Sequence[str],
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build an OpenBookQA multiple-choice prompt."""

    choice_texts = _clean_choices(choices)
    answer_text = _clean_answer_choice(answer, choice_texts)
    blocks = [_format_obqa_example(example) for example in examples]
    blocks.append(
        _format_obqa_block(
            question=_clean_text(question, "question"),
            choices=choice_texts,
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=answer_text,
        choices=choice_texts,
        max_new_tokens=max_new_tokens,
    )


def build_race_prompt(
    passage: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a RACE passage-based multiple-choice prompt."""

    choice_texts = _clean_choices(choices)
    answer_text = _clean_answer_choice(answer, choice_texts)
    blocks = [_format_race_example(example) for example in examples]
    blocks.append(
        _format_race_block(
            passage=_clean_text(passage, "passage"),
            question=_clean_text(question, "question"),
            choices=choice_texts,
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=answer_text,
        choices=choice_texts,
        max_new_tokens=max_new_tokens,
    )


def build_hellaswag_prompt(
    context: str,
    choices: Sequence[str],
    answer: str,
    examples: Sequence[PromptExample] = (),
    *,
    activity_label: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PromptResult:
    """Build a HellaSwag commonsense continuation prompt."""

    choice_texts = _clean_choices(choices)
    answer_text = _clean_answer_choice(answer, choice_texts)
    blocks = [_format_hellaswag_example(example) for example in examples]
    blocks.append(
        _format_hellaswag_block(
            context=_clean_text(context, "context"),
            choices=choice_texts,
            activity_label=_optional_clean_text(activity_label, "activity_label"),
        )
    )
    return _prompt_result(
        prompt="\n\n".join(blocks),
        answer=answer_text,
        choices=choice_texts,
        max_new_tokens=max_new_tokens,
    )


def _format_lambada_block(context: str, answer: str | None = None) -> str:
    lines = ["LAMBADA", f"Context: {context}"]
    lines.append("Completion:" if answer is None else f"Completion: {answer}")
    return "\n".join(lines)


def _format_mmlu_example(example: PromptExample) -> str:
    choices = _required_choices(example)
    return _format_mmlu_block(
        question=_required_text(example, "question"),
        choices=choices,
        answer=_clean_answer_choice(_required_text(example, "answer"), choices),
        subject=_optional_example_text(example, "subject"),
    )


def _format_mmlu_block(
    *,
    question: str,
    choices: tuple[str, ...],
    answer: str | None = None,
    subject: str | None = None,
) -> str:
    fields = []
    if subject is not None:
        fields.append(("Subject", subject))
    fields.append(("Question", question))
    return _format_multiple_choice_block(
        "MMLU",
        fields=fields,
        choices=choices,
        answer=answer,
    )


def _format_siqa_example(example: PromptExample) -> str:
    choices = _required_choices(example)
    return _format_siqa_block(
        context=_required_text(example, "context"),
        question=_required_text(example, "question"),
        choices=choices,
        answer=_clean_answer_choice(_required_text(example, "answer"), choices),
    )


def _format_siqa_block(
    *,
    context: str,
    question: str,
    choices: tuple[str, ...],
    answer: str | None = None,
) -> str:
    return _format_multiple_choice_block(
        "Social IQA",
        fields=(("Context", context), ("Question", question)),
        choices=choices,
        answer=answer,
    )


def _format_squad_example(example: PromptExample) -> str:
    return _format_squad_block(
        context=_required_text(example, "context"),
        question=_required_text(example, "question"),
        answer=_required_text(example, "answer"),
    )


def _format_squad_block(
    *,
    context: str,
    question: str,
    answer: str | None = None,
) -> str:
    lines = ["SQuAD", f"Context: {context}", f"Question: {question}"]
    lines.append("Answer:" if answer is None else f"Answer: {answer}")
    return "\n".join(lines)


def _format_story_cloze_example(example: PromptExample) -> str:
    choices = _required_choices(example)
    return _format_story_cloze_block(
        story=_required_text(example, "story"),
        choices=choices,
        answer=_clean_answer_choice(_required_text(example, "answer"), choices),
    )


def _format_story_cloze_block(
    *,
    story: str,
    choices: tuple[str, ...],
    answer: str | None = None,
) -> str:
    return _format_multiple_choice_block(
        "Story Cloze",
        fields=(("Story", story),),
        choices=choices,
        answer=answer,
    )


def _format_obqa_example(example: PromptExample) -> str:
    choices = _required_choices(example)
    return _format_obqa_block(
        question=_required_text(example, "question"),
        choices=choices,
        answer=_clean_answer_choice(_required_text(example, "answer"), choices),
    )


def _format_obqa_block(
    *,
    question: str,
    choices: tuple[str, ...],
    answer: str | None = None,
) -> str:
    return _format_multiple_choice_block(
        "OpenBookQA",
        fields=(("Question", question),),
        choices=choices,
        answer=answer,
    )


def _format_race_example(example: PromptExample) -> str:
    choices = _required_choices(example)
    return _format_race_block(
        passage=_required_text(example, "passage"),
        question=_required_text(example, "question"),
        choices=choices,
        answer=_clean_answer_choice(_required_text(example, "answer"), choices),
    )


def _format_race_block(
    *,
    passage: str,
    question: str,
    choices: tuple[str, ...],
    answer: str | None = None,
) -> str:
    return _format_multiple_choice_block(
        "RACE",
        fields=(("Passage", passage), ("Question", question)),
        choices=choices,
        answer=answer,
    )


def _format_hellaswag_example(example: PromptExample) -> str:
    choices = _required_choices(example)
    return _format_hellaswag_block(
        context=_required_text(example, "context"),
        choices=choices,
        answer=_clean_answer_choice(_required_text(example, "answer"), choices),
        activity_label=_optional_example_text(example, "activity_label"),
    )


def _format_hellaswag_block(
    *,
    context: str,
    choices: tuple[str, ...],
    answer: str | None = None,
    activity_label: str | None = None,
) -> str:
    fields = []
    if activity_label is not None:
        fields.append(("Activity", activity_label))
    fields.append(("Context", context))
    return _format_multiple_choice_block(
        "HellaSwag",
        fields=fields,
        choices=choices,
        answer=answer,
    )


def _format_multiple_choice_block(
    heading: str,
    *,
    fields: Sequence[tuple[str, str]],
    choices: tuple[str, ...],
    answer: str | None = None,
) -> str:
    lines = [heading]
    lines.extend(f"{label}: {value}" for label, value in fields)
    lines.append("Options:")
    lines.extend(
        f"{label}. {choice}"
        for label, choice in zip(_choice_labels(len(choices)), choices)
    )
    lines.append("Answer:" if answer is None else f"Answer: {answer}")
    return "\n".join(lines)


def _choice_labels(count: int) -> tuple[str, ...]:
    if count > 26:
        raise ValueError("choices must contain no more than 26 options")
    return tuple(chr(ord("A") + index) for index in range(count))


def _required_text(example: PromptExample, field: str) -> str:
    if field not in example:
        raise ValueError(f"example is missing required field {field!r}")
    return _clean_text(example[field], field)


def _optional_example_text(example: PromptExample, field: str) -> str | None:
    if field not in example:
        return None
    return _clean_text(example[field], field)


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _optional_clean_text(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field)


def _required_choices(example: PromptExample) -> tuple[str, ...]:
    if "choices" not in example:
        raise ValueError("example is missing required field 'choices'")
    return _clean_choices(example["choices"])


def _clean_choices(choices: object) -> tuple[str, ...]:
    if isinstance(choices, str) or not isinstance(choices, Sequence):
        raise ValueError("choices must be an ordered sequence of strings")
    choice_texts = tuple(_clean_text(choice, "choice") for choice in choices)
    if not choice_texts:
        raise ValueError("choices must contain at least one option")
    _choice_labels(len(choice_texts))
    return choice_texts


def _clean_answer_choice(answer: object, choices: tuple[str, ...]) -> str:
    answer_text = _clean_text(answer, "answer")
    if answer_text not in choices:
        label_to_choice = dict(zip(_choice_labels(len(choices)), choices))
        answer_label = answer_text.upper()
        if answer_label in label_to_choice:
            return label_to_choice[answer_label]
        raise ValueError(
            "answer must match a canonical choice text or option label"
        )
    return answer_text


def _prompt_result(
    *,
    prompt: str,
    answer: str,
    choices: tuple[str, ...] = (),
    max_new_tokens: int,
) -> PromptResult:
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    return PromptResult(
        prompt=prompt,
        answer=answer,
        choices=choices,
        max_new_tokens=max_new_tokens,
    )


__all__ = (
    "PromptResult",
    "build_lambada_prompt",
    "build_mmlu_prompt",
    "build_siqa_prompt",
    "build_squad_prompt",
    "build_story_cloze_prompt",
    "build_obqa_prompt",
    "build_race_prompt",
    "build_hellaswag_prompt",
)
