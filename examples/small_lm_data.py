"""Story-level training data for small LM alignment fine-tuning.

Every training window starts at a story boundary (one story per window), so
the model always sees a story opening in context. Pair samples additionally
mask the opening targets, training pure "given an opening, continue the
story" behaviour instead of spending capacity predicting the opening itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from examples.small_lm_text import clean_text_artifacts

DEDUP_PREFIX_TOKENS = 8
MIN_OPENING_WORDS = 3


def split_opening(text: str) -> tuple[str, str]:
    """Split story text into (opening, rest) at a sentence boundary.

    The opening is the first sentence; when it carries fewer than
    ``MIN_OPENING_WORDS`` words the opening extends to the second boundary.
    """
    boundaries = [
        i
        for i, ch in enumerate(text)
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1].isspace())
    ]
    if not boundaries:
        return text, ""
    end = boundaries[0]
    if len(text[: end + 1].split()) < MIN_OPENING_WORDS and len(boundaries) > 1:
        end = boundaries[1]
    return text[: end + 1], text[end + 1 :]


@dataclass(frozen=True)
class StoryTokens:
    tokens: list[int]
    opening_len: int
    pair_ok: bool


def tokenize_story(
    text: str,
    tokenizer,
    eos_token_id: int | None,
    append_eos: bool,
    clean_text: bool = True,
) -> StoryTokens:
    if clean_text:
        text = clean_text_artifacts(text)
    opening, rest = split_opening(text)
    opening_ids = tokenizer.encode(opening, add_special_tokens=False)
    rest_ids = tokenizer.encode(rest, add_special_tokens=False)
    tokens = [*opening_ids, *rest_ids]
    if append_eos and eos_token_id is not None:
        tokens.append(eos_token_id)
    return StoryTokens(
        tokens=tokens,
        opening_len=len(opening_ids),
        pair_ok=bool(rest_ids),
    )


def collect_stories(
    hf_dataset,
    tokenizer,
    max_tokens: int,
    text_field: str,
    eos_token_id: int | None,
    append_eos: bool,
    clean_text: bool = True,
) -> list[StoryTokens]:
    if append_eos and eos_token_id is None:
        raise ValueError(
            "append_eos=True but tokenizer has no eos_token_id; "
            "pass --no-append-eos or choose a tokenizer with EOS"
        )
    stories: list[StoryTokens] = []
    total = 0
    for sample in hf_dataset:
        text = sample[text_field]
        if isinstance(text, str) and text.strip():
            story = tokenize_story(text, tokenizer, eos_token_id, append_eos, clean_text)
            if story.tokens:
                stories.append(story)
                total += len(story.tokens)
        if total >= max_tokens:
            break
    if total <= 1:
        raise ValueError(f"dataset produced only {total} tokens; need > 1")
    return stories


def dedup_stories(
    stories: list[StoryTokens], enabled: bool
) -> tuple[list[StoryTokens], int]:
    """Drop template stories: exact duplicates and shared 8-token openings."""
    if not enabled:
        return list(stories), 0
    kept: list[StoryTokens] = []
    dropped = 0
    seen_full: set[tuple[int, ...]] = set()
    seen_prefix: set[tuple[int, ...]] = set()
    for story in stories:
        key = tuple(story.tokens)
        prefix = key[:DEDUP_PREFIX_TOKENS]
        if key in seen_full or prefix in seen_prefix:
            dropped += 1
            continue
        seen_full.add(key)
        seen_prefix.add(prefix)
        kept.append(story)
    return kept, dropped


class StoryAlignedDataset(Dataset):
    """One fixed-length window per story, padded and masked.

    ``pair_mix`` is the deterministic fraction of pair-eligible stories whose
    opening targets are masked (Bresenham-style selection). Padding targets
    are always masked. Every sample keeps at least one unmasked target.
    """

    def __init__(
        self,
        stories: list[StoryTokens],
        seq_len: int,
        pair_mix: float,
        eos_token_id: int | None,
    ) -> None:
        if not 0.0 <= pair_mix <= 1.0:
            raise ValueError(f"pair_mix must be within [0, 1], got {pair_mix}")
        if eos_token_id is None:
            raise ValueError("story packing needs eos_token_id for padding")
        self.seq_len = seq_len
        self.windows: list[list[int]] = []
        self.masks: list[list[float]] = []
        pair_acc = 0.0
        for story in stories:
            content_len = min(len(story.tokens), seq_len + 1)
            if content_len < 2:
                continue
            use_pair = False
            if story.pair_ok and story.opening_len <= seq_len:
                pair_acc += pair_mix
                if pair_acc >= 1.0:
                    pair_acc -= 1.0
                    use_pair = True
            window = story.tokens[:content_len]
            window.extend([eos_token_id] * (seq_len + 1 - content_len))
            mask = [1.0] * self.seq_len
            if use_pair and story.opening_len > 1:
                mask[: story.opening_len - 1] = [0.0] * (story.opening_len - 1)
            if content_len <= seq_len:
                mask[content_len - 1 :] = [0.0] * (seq_len - content_len + 1)
            self.windows.append(window)
            self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = torch.tensor(self.windows[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return data[:-1], data[1:], mask
