# TinyStories v6.1 Seq384 Natural-Stop Recovery Runbook

v6 lowered validation loss, but the fixed 60-sample prompt-file evaluation
regressed from v5's `60/60` EOS stops to `58/60`. Both failures were the same
prompt, `Lily found a small door in the garden`, and both stopped because the
256-token context was exhausted.

This runbook turns that diagnosis into a v6.1 plan: expand context to 384,
warm-start from v6, and run a short continuation to recover natural stopping
without giving up v6's lower validation loss.

## Diagnosis Already Observed

A no-training context resize probe was created with:

```bash
python -m examples.resize_small_lm_context \
  --checkpoint checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt \
  --out /tmp/openmythos_tinystories_v6_seq384_probe.pt \
  --max-seq-len 384
```

Result:

```text
[resized] ... max_seq_len=256->384 loaded=66 skipped=freqs_cis
```

The only skipped key was `freqs_cis`, which is expected because RoPE frequencies
must be rebuilt for the longer context.

The two v6 failure cases then reached EOS with the seq384 probe:

| Prompt | Preset | Seed | v6 seq256 result | seq384 probe result |
|---|---|---:|---|---|
| Lily found a small door in the garden | conservative | 207 | `max_seq_len` at 248 generated tokens | `eos` at 269 generated tokens |
| Lily found a small door in the garden | balanced | 208 | `max_seq_len` at 248 generated tokens | `eos` at 300 generated tokens |

This means the v6 failure is primarily a context/budget issue, not an absence of
EOS behavior. However, the long Lily samples become semantically weak near the
end, so the final v6.1 target is not merely "allow longer generations". The goal
is to make the model stop cleanly in a longer context while keeping output
quality acceptable.

## Preflight

```bash
cd /Users/macos/Desktop/Claude/OpenMythos
source .venv-arm64/bin/activate

ls -lh checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt
```

## Optional No-Training Probe

Use this probe to verify context expansion still works before training:

```bash
python -m examples.resize_small_lm_context \
  --checkpoint checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt \
  --out /tmp/openmythos_tinystories_v6_seq384_probe.pt \
  --max-seq-len 384

python -m examples.generate_small_lm \
  --checkpoint /tmp/openmythos_tinystories_v6_seq384_probe.pt \
  --prompt "Lily found a small door in the garden" \
  --device mps \
  --max-new-tokens 300 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 0 \
  --seed 208
```

Expected diagnostic signal: `eos_hit=True`, likely after more than 248 generated
tokens. If this does not happen, pause before training v6.1.

## Smoke Train

Use a conservative batch size first. Seq384 uses more memory than seq256, and
`batch-size=4` keeps the MPS run safer.

```bash
python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories \
  --dataset-config "" \
  --init-from checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt \
  --out /tmp/openmythos_tinystories_v61_seq384_smoke.pt \
  --steps 300 \
  --batch-size 4 \
  --seq-len 384 \
  --device mps \
  --lr 5e-5 \
  --n-loops 4 \
  --log-every 50 \
  --eval-every 150 \
  --eval-batches 20 \
  --save-every 300 \
  --train-tokens 1000000 \
  --eval-tokens 75000 \
  --eos-loss-weight 4.0
```

Proceed if:

- The setup line prints `seq_len=384` and `clean_text=True`
- `[init-from]` starts at step `100000`
- The init load reports only `freqs_cis` skipped
- Loss remains finite
- Validation loss is roughly in the v6 neighborhood, not a sharp regression

## Full v6.1 Run

```bash
python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories \
  --dataset-config "" \
  --init-from checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt \
  --out checkpoints/tinystories_mps_small_mythos_seq384_clean_v61.pt \
  --steps 5000 \
  --batch-size 4 \
  --seq-len 384 \
  --device mps \
  --lr 5e-5 \
  --n-loops 4 \
  --log-every 50 \
  --eval-every 500 \
  --eval-batches 20 \
  --save-every 1000 \
  --train-tokens 20000000 \
  --eval-tokens 200000 \
  --eos-loss-weight 4.0
```

This should produce a checkpoint at step `105000` because `--init-from` carries
the source step forward and adds `5000` new steps.

## Evaluation Gates

Primary recovery evaluation, allowing the longer context to matter:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/tinystories_mps_small_mythos_seq384_clean_v61.pt \
  --device mps \
  --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --max-new-tokens 300 \
  --stop-at-eos \
  --out-dir runs
```

Strict comparison evaluation, using the old 256-token generation budget:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/tinystories_mps_small_mythos_seq384_clean_v61.pt \
  --device mps \
  --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --max-new-tokens 256 \
  --stop-at-eos \
  --out-dir runs
```

Interpretation:

- Success: `60/60` EOS at `max-new-tokens=300`, validation loss still close to
  or better than v6.
- Strong success: `60/60` EOS at `max-new-tokens=256`, meaning v6.1 pulled the
  long-tail Lily failures earlier rather than only expanding context.
- If seq384 reaches `60/60` only at 300 tokens, archive it as a long-context
  recovery model, not a strict v5 replacement.

## Archive Checklist

Archive only after evaluation:

```bash
cp checkpoints/tinystories_mps_small_mythos_seq384_clean_v61.pt \
  checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt

ls -lh checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt
shasum -a 256 checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt
```

Then add a model card recording both the 300-token recovery score and the strict
256-token score.
