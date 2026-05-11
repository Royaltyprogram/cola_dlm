import pytest

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
)


def test_evaluation_public_surface():
    import cola_dlm.evaluation as evaluation

    assert evaluation.__all__ == (
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


def test_prompt_result_is_a_small_metadata_container():
    result = PromptResult(prompt="Answer:", answer="yes")

    assert result.prompt == "Answer:"
    assert result.answer == "yes"
    assert result.choices == ()
    assert result.max_new_tokens == 32


def test_lambada_prompt_formats_final_completion_and_uses_default_tokens():
    result = build_lambada_prompt(
        "The explorer found a hidden",
        "map",
        examples=[
            {
                "context": "Marta reached for the bright red",
                "answer": "apple",
            }
        ],
    )

    assert result.prompt == (
        "LAMBADA\n"
        "Context: Marta reached for the bright red\n"
        "Completion: apple\n\n"
        "LAMBADA\n"
        "Context: The explorer found a hidden\n"
        "Completion:"
    )
    assert result.answer == "map"
    assert result.choices == ()
    assert result.max_new_tokens == 32


def test_squad_prompt_uses_context_question_and_answer_cue_without_choices():
    result = build_squad_prompt(
        context="Ada wrote the notes in London.",
        question="Where did Ada write the notes?",
        answer="London",
        examples=[
            {
                "context": "Ben packed the blue bag.",
                "question": "Which bag did Ben pack?",
                "answer": "the blue bag",
            }
        ],
        max_new_tokens=9,
    )

    assert result.prompt == (
        "SQuAD\n"
        "Context: Ben packed the blue bag.\n"
        "Question: Which bag did Ben pack?\n"
        "Answer: the blue bag\n\n"
        "SQuAD\n"
        "Context: Ada wrote the notes in London.\n"
        "Question: Where did Ada write the notes?\n"
        "Answer:"
    )
    assert result.answer == "London"
    assert result.choices == ()
    assert result.max_new_tokens == 9


def test_requested_benchmark_builders_accept_fake_examples():
    mmlu = build_mmlu_prompt(
        question="Which organ pumps blood?",
        choices=["heart", "lung", "skin", "bone"],
        answer="heart",
        examples=[
            {
                "subject": "biology",
                "question": "Which gas do plants release?",
                "choices": ["oxygen", "helium", "argon", "neon"],
                "answer": "oxygen",
            }
        ],
        subject="biology",
    )
    siqa = build_siqa_prompt(
        context="Riley dropped the glass.",
        question="What will Riley likely do next?",
        choices=["apologize", "sleep", "paint"],
        answer="apologize",
        examples=[
            {
                "context": "Avery missed the bus.",
                "question": "What might Avery need?",
                "choices": ["a ride", "a cake", "a hat"],
                "answer": "a ride",
            }
        ],
    )
    story_cloze = build_story_cloze_prompt(
        story="Nia trained every morning. The race began at noon.",
        choices=["She finished strong.", "She forgot how to run."],
        answer="She finished strong.",
        examples=[
            {
                "story": "Tom mixed flour and water. He put the pan in the oven.",
                "choices": ["Bread came out.", "The moon rose."],
                "answer": "Bread came out.",
            }
        ],
    )
    obqa = build_obqa_prompt(
        question="What do plants need for photosynthesis?",
        choices=["sunlight", "sand", "plastic", "iron filings"],
        answer="sunlight",
        examples=[
            {
                "question": "What protects Earth from many UV rays?",
                "choices": ["ozone layer", "paper", "salt", "wax"],
                "answer": "ozone layer",
            }
        ],
    )
    race = build_race_prompt(
        passage="The train left after the bell rang.",
        question="What happened first?",
        choices=["The bell rang.", "The train left.", "The train stopped."],
        answer="The bell rang.",
        examples=[
            {
                "passage": "Lena read the sign before entering.",
                "question": "What did Lena do first?",
                "choices": ["read the sign", "entered", "slept"],
                "answer": "read the sign",
            }
        ],
    )
    hellaswag = build_hellaswag_prompt(
        context="A chef cracks eggs into a bowl and",
        choices=["whisks them together.", "parks a car.", "reads a map."],
        answer="whisks them together.",
        examples=[
            {
                "activity_label": "making tea",
                "context": "A person pours hot water over leaves and",
                "choices": ["lets them steep.", "throws a ball.", "ties shoes."],
                "answer": "lets them steep.",
            }
        ],
        activity_label="cooking",
    )

    assert "MMLU\nSubject: biology" in mmlu.prompt
    assert "Question: Which organ pumps blood?" in mmlu.prompt
    assert "Social IQA\nContext: Avery missed the bus." in siqa.prompt
    assert "Question: What will Riley likely do next?" in siqa.prompt
    assert "Story Cloze\nStory: Tom mixed flour and water." in story_cloze.prompt
    assert "Story: Nia trained every morning." in story_cloze.prompt
    assert "OpenBookQA\nQuestion: What protects Earth" in obqa.prompt
    assert "Question: What do plants need for photosynthesis?" in obqa.prompt
    assert "RACE\nPassage: Lena read the sign before entering." in race.prompt
    assert "Question: What happened first?" in race.prompt
    assert "HellaSwag\nActivity: making tea" in hellaswag.prompt
    assert "Activity: cooking" in hellaswag.prompt


def test_fixed_examples_appear_before_query_and_preserve_caller_order():
    result = build_obqa_prompt(
        question="Query question",
        choices=["query answer", "query distractor"],
        answer="query answer",
        examples=[
            {
                "question": "First fixed question",
                "choices": ["first answer", "first distractor"],
                "answer": "first answer",
            },
            {
                "question": "Second fixed question",
                "choices": ["second answer", "second distractor"],
                "answer": "second answer",
            },
        ],
    )

    assert result.prompt.index("First fixed question") < result.prompt.index(
        "Second fixed question"
    )
    assert result.prompt.index("Second fixed question") < result.prompt.index(
        "Query question"
    )


def test_multiple_choice_results_store_option_text_separately_from_labels():
    result = build_mmlu_prompt(
        question="Which choice is canonical?",
        choices=["alpha text", "beta text", "gamma text"],
        answer="beta text",
    )

    assert "A. alpha text" in result.prompt
    assert "B. beta text" in result.prompt
    assert result.choices == ("alpha text", "beta text", "gamma text")
    assert result.answer == "beta text"
    assert all(not choice.startswith(("A.", "B.", "C.")) for choice in result.choices)


def test_multiple_choice_answer_label_is_converted_to_canonical_option_text():
    result = build_obqa_prompt(
        question="Which object is hot?",
        choices=["fire", "ice"],
        answer="A",
    )

    assert result.answer == "fire"
    assert result.choices == ("fire", "ice")


def test_max_new_tokens_can_be_overridden_on_multiple_choice_prompts():
    result = build_race_prompt(
        passage="The class ended when the bell rang.",
        question="Why did class end?",
        choices=["the bell rang", "it snowed"],
        answer="the bell rang",
        max_new_tokens=3,
    )

    assert result.max_new_tokens == 3


def test_prompt_builders_reject_invalid_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens must be a positive"):
        build_lambada_prompt("The final", "word", max_new_tokens=0)
