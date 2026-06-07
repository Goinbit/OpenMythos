import torch

from examples.generate_small_lm import (
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    summarize_generation_stop,
    trim_after_sentences,
)


def test_apply_repetition_penalty_only_changes_seen_tokens():
    logits = torch.tensor([[2.0, -4.0, 1.5, 0.5]])
    generated = torch.tensor([[0, 1, 0]])

    adjusted = apply_repetition_penalty(logits, generated, penalty=2.0)

    torch.testing.assert_close(adjusted[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(adjusted[0, 1], torch.tensor(-8.0))
    torch.testing.assert_close(adjusted[0, 2], logits[0, 2])
    torch.testing.assert_close(adjusted[0, 3], logits[0, 3])
    torch.testing.assert_close(logits, torch.tensor([[2.0, -4.0, 1.5, 0.5]]))


def test_apply_no_repeat_ngram_bans_seen_continuations():
    logits = torch.zeros(1, 6)
    generated = torch.tensor([[1, 2, 3, 1, 2]])

    adjusted = apply_no_repeat_ngram(logits, generated, ngram_size=3)

    assert adjusted[0, 3].isneginf()
    torch.testing.assert_close(adjusted[0, 0], logits[0, 0])
    torch.testing.assert_close(adjusted[0, 4], logits[0, 4])


def test_apply_no_repeat_ngram_size_one_bans_seen_tokens_per_batch():
    logits = torch.zeros(2, 5)
    generated = torch.tensor([[1, 2], [3, 4]])

    adjusted = apply_no_repeat_ngram(logits, generated, ngram_size=1)

    assert adjusted[0, 1].isneginf()
    assert adjusted[0, 2].isneginf()
    assert adjusted[1, 3].isneginf()
    assert adjusted[1, 4].isneginf()
    torch.testing.assert_close(adjusted[0, 3], logits[0, 3])
    torch.testing.assert_close(adjusted[1, 1], logits[1, 1])


def test_trim_after_sentences_counts_after_prompt_prefix():
    text = "Prompt already. First generated. Second generated! Third stays."

    trimmed = trim_after_sentences(text, sentence_count=2, start_offset=15)

    assert trimmed == "Prompt already. First generated. Second generated!"


def test_trim_after_sentences_keeps_text_without_enough_sentences():
    text = "Prompt. One generated sentence."

    assert trim_after_sentences(text, sentence_count=3, start_offset=8) == text


def test_summarize_generation_stop_detects_eos():
    output_ids = torch.tensor([10, 11, 12, 99, 13])

    summary = summarize_generation_stop(
        output_ids,
        prompt_len=2,
        max_new_tokens=20,
        max_seq_len=64,
        eos_token_id=99,
        stop_at_eos=True,
    )

    assert summary == (True, 3, "eos")


def test_summarize_generation_stop_reports_max_seq_len_before_budget():
    output_ids = torch.full((256,), 1)

    summary = summarize_generation_stop(
        output_ids,
        prompt_len=8,
        max_new_tokens=400,
        max_seq_len=256,
        eos_token_id=99,
        stop_at_eos=True,
    )

    assert summary == (False, 248, "max_seq_len")
