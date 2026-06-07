"""Generate text from a checkpoint produced by examples.train_small_lm."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from open_mythos import MythosConfig, OpenMythos, StaticKVCache

SENTENCE_ENDINGS = {".", "!", "?"}
CLOSING_PUNCTUATION = {'"', "'", "”", "’", ")", "]"}


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError(f"--repetition-penalty must be positive, got {penalty}")
    adjusted = logits.clone()
    if penalty == 1.0:
        return adjusted
    for batch_idx in range(generated_ids.shape[0]):
        seen = generated_ids[batch_idx].unique()
        token_scores = adjusted[batch_idx, seen]
        adjusted[batch_idx, seen] = torch.where(
            token_scores < 0,
            token_scores * penalty,
            token_scores / penalty,
        )
    return adjusted


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    adjusted = logits.clone()
    if ngram_size <= 0:
        return adjusted
    if ngram_size == 1:
        for batch_idx in range(generated_ids.shape[0]):
            adjusted[batch_idx, generated_ids[batch_idx].unique()] = float("-inf")
        return adjusted

    prefix_len = ngram_size - 1
    for batch_idx in range(generated_ids.shape[0]):
        tokens = generated_ids[batch_idx].tolist()
        if len(tokens) < prefix_len:
            continue
        current_prefix = tuple(tokens[-prefix_len:])
        banned: list[int] = []
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tokens[i : i + ngram_size]
            if tuple(ngram[:-1]) == current_prefix:
                banned.append(ngram[-1])
        if banned:
            adjusted[batch_idx, torch.tensor(banned, device=logits.device)] = float(
                "-inf"
            )
    return adjusted


def trim_after_sentences(
    text: str,
    sentence_count: int,
    start_offset: int = 0,
) -> str:
    if sentence_count <= 0:
        return text

    seen = 0
    i = max(0, start_offset)
    while i < len(text):
        if text[i] in SENTENCE_ENDINGS:
            seen += 1
            if seen >= sentence_count:
                end = i + 1
                while end < len(text) and text[end] in CLOSING_PUNCTUATION:
                    end += 1
                return text[:end].rstrip()
        i += 1
    return text


@torch.no_grad()
def generate_tokens(
    model: OpenMythos,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    n_loops: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: int | None = None,
    stop_at_eos: bool = False,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"--temperature must be positive, got {temperature}")

    kv_cache = StaticKVCache(max_seq_len=model.cfg.max_seq_len)
    prompt_len = input_ids.shape[1]
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for step in range(max_new_tokens):
        if step == 0:
            cur_ids = input_ids
            start_pos = 0
        else:
            cur_ids = input_ids[:, -1:]
            start_pos = prompt_len + step - 1
        if start_pos + cur_ids.shape[1] > model.cfg.max_seq_len:
            break

        logits = model(cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
        logits = logits[:, -1, :] / temperature
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, input_ids, no_repeat_ngram_size)
        if top_k > 0:
            v, _ = logits.topk(min(top_k, logits.shape[-1]))
            logits[logits < v[:, -1:]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        if stop_at_eos and eos_token_id is not None:
            next_tok = torch.where(
                finished[:, None],
                torch.full_like(next_tok, eos_token_id),
                next_tok,
            )
            finished |= next_tok.squeeze(1) == eos_token_id
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if stop_at_eos and bool(finished.all()):
            break
    return input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/small_mythos.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument(
        "--stop-after-sentences",
        type=int,
        default=0,
        help="trim decoded output after this many generated sentences; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run examples/train_small_lm.py first."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer_id = ckpt.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    cfg = MythosConfig(**ckpt["config"])
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("--prompt must encode to at least one token")
    prompt_ids = prompt_ids[-max(1, cfg.max_seq_len - 1):]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

    with torch.no_grad():
        out = generate_tokens(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            n_loops=args.n_loops,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            stop_at_eos=args.stop_at_eos,
        )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    text = trim_after_sentences(
        text,
        sentence_count=args.stop_after_sentences,
        start_offset=len(prompt_text),
    )
    print(f"[loaded] {path} step={ckpt.get('step')} loss={ckpt.get('loss'):.4f}")
    print(
        "[generate] "
        f"temperature={args.temperature} top_k={args.top_k} "
        f"repetition_penalty={args.repetition_penalty} "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size} "
        f"stop_after_sentences={args.stop_after_sentences}"
    )
    print(text)


if __name__ == "__main__":
    main()
