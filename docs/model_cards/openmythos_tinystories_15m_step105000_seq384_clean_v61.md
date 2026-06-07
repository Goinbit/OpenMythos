# OpenMythos TinyStories 15M Step 105000 Seq384 Clean v6.1

This is the eighth archived local OpenMythos small language model trained on
Apple Silicon MPS. It warm-starts from the step 100000 seq256 clean v6
checkpoint, expands context length to 384, and runs a short 5k-step continuation
focused on recovering v6's natural-stop regression.

v6.1 is the current best long-context TinyStories checkpoint in this local run:
it keeps v6's lower validation-loss direction, improves strict 256-token EOS
reliability from v6's 58/60 to 59/60, and reaches 60/60 EOS when evaluated with
the full seq384 generation budget.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt`
- Size: `182M`
- SHA256: `9caf6ce9a4275122a34d5359905684f39e6aa13a6098d77551db6437a993638d`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_seq384_clean_v61.pt`
- Warm-start source: `checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt`
- Tokenizer: `gpt2`
- `append_eos`: `True`
- `eos_loss_weight`: `4.0`
- `clean_text`: `True`

## Model

- Parameters: `15,856,162`
- Vocab size: `50,257`
- Hidden dim: `256`
- Attention: `gqa`
- Heads: `8`
- KV heads: `2`
- Max sequence length: `384`
- Prelude layers: `1`
- Recurrent loops: `4`
- Coda layers: `1`
- MoE experts: `8`
- Shared experts: `1`
- Experts per token: `2`
- Expert dim: `256`
- LoRA rank: `8`

## Training

- Dataset: `roneneldan/TinyStories`
- Dataset config: none
- Device: Apple Silicon MPS
- Environment: `.venv-arm64`
- Python: arm64 Python 3.12
- Step: `105000`
- Final train loss: `1.8100`
- Validation loss: `1.6227`
- Observed late-run throughput: about `4.97k tok/s`
- Continuation mode: `--init-from` v6 with fresh optimizer
- Learning rate: `5e-5`
- Batch size: `4`
- Sequence length: `384`
- EOS handling: tokenizer EOS appended after each text sample; EOS target loss
  multiplied by `4.0`.
- Text hygiene: `examples.train_small_lm` cleaned common mojibake and spacing
  artifacts before tokenization.

Smoke run before the full continuation reached validation loss `1.6335` at step
`100200`, confirming seq384 warm-start was healthy.

Comparison with recent baselines:

| Checkpoint | Step | Max seq len | Validation loss | EOS @256 | EOS @300 | EOS @full budget |
|---|---:|---:|---:|---:|---:|---:|
| Seq256 EOS weighted v5 | 90000 | 256 | 1.7513 | 60/60 | not measured | 60/60 @256 |
| Seq256 clean v6 | 100000 | 256 | 1.6520 | 58/60 | not measured | 58/60 @256 |
| Seq384 clean v6.1 | 105000 | 384 | 1.6227 | 59/60 | 59/60 | 60/60 @376 |

## Recommended Generation

Balanced mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt \
  --prompt "Once upon a time" \
  --device mps \
  --max-new-tokens 300 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 0
```

For polished demos, add `--stop-after-sentences 3` or `4`. For natural-stop
measurement, keep `--stop-after-sentences 0`.

## Evaluation Snapshot

Latest local reports:

- `runs/generation_eval_1780815382.md` (`max-new-tokens=300`)
- `runs/generation_eval_1780815564.md` (`max-new-tokens=256`, strict comparison)
- `runs/generation_eval_1780815803.md` (`max-new-tokens=376`, full seq384 budget)

Natural-stop summary:

| Budget | Samples | EOS hits | Stop reasons |
|---:|---:|---:|---|
| 256 | 60 | 59/60 | `eos=59`, `max_new_tokens=1` |
| 300 | 60 | 59/60 | `eos=59`, `max_new_tokens=1` |
| 376 | 60 | 60/60 | `eos=60` |

The remaining 256/300-token failure is the conservative Lily sample:

- Prompt: `Lily found a small door in the garden`
- Preset: conservative (`temperature=0.55`, `top_k=15`)
- Seed: `207`
- At 256 tokens: `stop_reason=max_new_tokens`
- At 300 tokens: `stop_reason=max_new_tokens`
- At 376 tokens: `eos_hit=True`, `generated_tokens=364`, `stop_reason=eos`

Representative sample:

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, she found a shiny rock and decided to keep it. She put it in her pocket and ran home to show her mommy.

Representative long-tail sample:

> Lily found a small door in the garden. She opened it and saw some people inside. They were wearing colorful clothes and had a big bag. Lily was curious and went to see what they were doing.
>
> She saw a lady with a big hat and a black dress. The lady looked at her and smiled. She said, "Hello, little girl. Do you like my house?"

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- v6.1 recovers 60/60 EOS only with the full seq384 generation budget; strict
  256-token and 300-token evaluations remain 59/60.
- The long-tail Lily conservative sample requires 364 generated tokens to reach
  EOS, and long continuations can become semantically weak.
- Some generated stories are still repetitive or logically inconsistent.
- The model has no instruction tuning, preference tuning, safety tuning, or
  factual grounding.

## Status

Use v6.1 when you want the lowest validation loss and the best long-context
natural-stop recovery. Use v5 when you need the strongest strict 256-token EOS
baseline. Use v6 only as the seq256 clean-text continuation reference point.
