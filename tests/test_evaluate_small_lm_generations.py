from examples.evaluate_small_lm_generations import (
    DEFAULT_PRESETS,
    DEFAULT_PROMPTS,
    GenerationResult,
    load_prompts,
    render_report,
)


def test_default_eval_suite_covers_30_to_60_samples():
    sample_count = len(DEFAULT_PROMPTS) * len(DEFAULT_PRESETS)

    assert len(DEFAULT_PROMPTS) >= 10
    assert len(DEFAULT_PRESETS) == 3
    assert 30 <= sample_count <= 60


def test_load_prompts_reads_non_empty_non_comment_lines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n".join(
            [
                "# TinyStories eval prompts",
                "",
                "Mia heard a bell in the forest",
                "  Sam built a boat from paper  ",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prompts = load_prompts(str(prompt_file))

    assert prompts == [
        "Mia heard a bell in the forest",
        "Sam built a boat from paper",
    ]


def test_render_report_includes_checkpoint_metadata_and_samples():
    results = [
        GenerationResult(
            prompt="Tom lost his red ball",
            text="Tom lost his red ball. He found it.",
            temperature=0.55,
            top_k=15,
            repetition_penalty=1.22,
            no_repeat_ngram_size=4,
            stop_after_sentences=3,
            stop_at_eos=True,
            eos_hit=True,
            generated_tokens=42,
            stop_reason="eos",
            seed=7,
            preset_name="conservative",
        ),
        GenerationResult(
            prompt="Lily found a door",
            text="Lily found a door. It was blue.",
            temperature=0.6,
            top_k=20,
            repetition_penalty=1.18,
            no_repeat_ngram_size=3,
            stop_after_sentences=4,
            stop_at_eos=True,
            eos_hit=False,
            generated_tokens=125,
            stop_reason="max_seq_len",
            seed=8,
            preset_name="balanced",
        ),
    ]

    report = render_report(
        checkpoint="checkpoints/model.pt",
        step=30000,
        loss=2.1267,
        results=results,
    )

    assert "# Small LM Generation Evaluation" in report
    assert "- checkpoint: `checkpoints/model.pt`" in report
    assert "- step: `30000`" in report
    assert "- loss: `2.1267`" in report
    assert "## Summary" in report
    assert "- samples: `2`" in report
    assert "- eos hit rate: `1/2` (`50.0%`)" in report
    assert "- stop reasons: `eos=1`, `max_seq_len=1`" in report
    assert "## Prompt Summary" in report
    assert "| Lily found a door | 0/1 | 0.0% | max_seq_len=1 |" in report
    assert "| Tom lost his red ball | 1/1 | 100.0% | eos=1 |" in report
    assert "## Preset Summary" in report
    assert "| balanced | 0/1 | 0.0% | max_seq_len=1 |" in report
    assert "| conservative | 1/1 | 100.0% | eos=1 |" in report
    assert "## Sample 1" in report
    assert "**prompt:** Tom lost his red ball" in report
    assert "temperature=0.55" in report
    assert "stop_at_eos=True" in report
    assert "eos_hit=True" in report
    assert "generated_tokens=42" in report
    assert "stop_reason=eos" in report
    assert "Tom lost his red ball. He found it." in report
