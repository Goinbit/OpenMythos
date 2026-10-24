# TinyStories v7 · 续写对齐训练(story packing + pair mix + dedup)

**目标:** 迭代计划 T1+T2([docs/small_lm_iteration_plan.md](../small_lm_iteration_plan.md))
**基线:** v6.1 `checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt`
**输出:** `checkpoints/tinystories_mps_small_mythos_seq384_story_v7.pt`(新文件,不覆盖)

---

## 0. Preflight

```bash
cd /Users/macos/Desktop/Claude/OpenMythos
.venv-arm64/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # True
ls checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt              # 存在
.venv-arm64/bin/python -m pytest tests/test_train_small_lm.py tests/test_evaluate_small_lm_generations.py -q  # 全绿
```

## 1. CPU 冒烟(2 步,验证新数据路径)

```bash
.venv-arm64/bin/python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --steps 2 --batch-size 2 --seq-len 32 --device cpu \
  --eval-every 0 --log-every 1 --save-every 2 \
  --train-tokens 5000 --eval-tokens 1000 \
  --packing story --pair-mix 0.5 --dedup-8gram --eos-loss-weight 4.0 \
  --seed 7 --out /tmp/openmythos_v7_story_smoke.pt
```

期望:`[setup] ... packing=story pair_mix=0.5 dedup_8gram=True`、loss 有限、checkpoint metadata 含 `packing/pair_mix/dedup_8gram`。

## 2. MPS 短冒烟(50 步,估吞吐)

```bash
.venv-arm64/bin/python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --steps 50 --batch-size 8 --seq-len 384 --device mps \
  --eval-every 50 --eval-batches 10 --log-every 10 --save-every 50 \
  --train-tokens 200000 --eval-tokens 20000 \
  --packing story --pair-mix 0.5 --dedup-8gram --eos-loss-weight 4.0 \
  --seed 7 --out /tmp/openmythos_v7_story_mps_smoke.pt
```

期望:loss 有限;用 tok/s 估算 5000 步总时长(5k tok/s ≈ 1 小时)。

## 3. 全量 v7 续训(5000 步)

```bash
.venv-arm64/bin/python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --init-from checkpoints/openmythos_tinystories_15m_step105000_seq384_clean_v61.pt \
  --steps 5000 --batch-size 8 --seq-len 384 --device mps \
  --lr 1e-4 --eos-loss-weight 4.0 \
  --packing story --pair-mix 0.5 --dedup-8gram \
  --eval-every 500 --eval-batches 20 --log-every 50 --save-every 500 \
  --seed 7 --out checkpoints/tinystories_mps_small_mythos_seq384_story_v7.pt
```

- 中断续跑:改用 `--resume checkpoints/tinystories_mps_small_mythos_seq384_story_v7.pt`(去掉 `--init-from`)
- 如 MPS OOM:`--batch-size 4`
- 数据语义注意:`--init-from` 只允许从**未记录 packing 字段**的旧 checkpoint(如 v6.1)切到新数据模式;一旦 v7 checkpoint 记录了 `packing=story`,后续 `--resume`/`--init-from` 必须保持相同 `--packing/--pair-mix/--dedup-8gram`,否则启动即拒绝

## 4. 验收评估(T1/T2 gate,一条命令 pass/fail)

```bash
.venv-arm64/bin/python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/tinystories_mps_small_mythos_seq384_story_v7.pt \
  --device mps \
  --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --preset-set coherence-v2 --max-new-tokens 256 --stop-at-eos \
  --require-eos 60/60 \
  --fail-on weak_prompt_adherence=15 \
  --fail-on missing_ending=15 \
  --fail-on repeated_trigram=5 \
  --out-dir runs
```

期望输出 `[gate] PASS` 且退出码 0;任一门槛未达标输出 `[gate] FAIL ...` 退出码 1。

## 5. 验收标准(对照迭代计划)

| 指标 | 基线 v6.1 | v7 目标 | 红线 |
|---|---|---|---|
| EOS | 60/60 | 60/60 | **不得回退** |
| weak_prompt_adherence | 29 | ≤ 15 | — |
| missing_ending | 27 | ≤ 15 | — |
| repeated_trigram | 14 | ≤ 5 | — |
| val loss(stream eval) | 1.6227 | ≤ 1.6427 | 回退 ≤ 0.02 |

## 6. 归档 checklist

- [ ] checkpoint 复制为 `checkpoints/openmythos_tinystories_15m_step<step>_seq384_story_v7.pt`
- [ ] `shasum -a 256` 记录
- [ ] 新建模型卡 `docs/model_cards/openmythos_tinystories_15m_step<step>_seq384_story_v7.md`
- [ ] 用 `examples.compare_small_lm_eval_reports --glob 'runs/generation_eval_*.md' --latest 3` 对比 v6.1 基线
- [ ] CSV 趋势:`--csv --out runs/latest_eval_compare.csv`
- [ ] 更新 AGENTS.md 执行日志
