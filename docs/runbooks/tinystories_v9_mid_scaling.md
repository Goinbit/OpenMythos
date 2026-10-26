# TinyStories v9 · T3b 扩容训练(small 15.8M → mid 37.7M)

**目标:** 迭代计划 T3b([docs/small_lm_iteration_plan.md](../small_lm_iteration_plan.md))
**变量隔离:** 数据配方与 v6.1 谱系完全一致(stream packing / clean text / EOS weight 4.0),
唯一变量是容量——验收 val < 1.55、endings ≥ 40/60 只归因于扩容。
**关键约束:** dim 变化后 `--init-from` 正确拒绝跨维 warm-start → **从头训练**。

---

## 0. Preflight

```bash
cd /Users/macos/Desktop/Claude/OpenMythos
.venv-arm64/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # True
.venv-arm64/bin/python -m pytest tests/test_train_small_lm.py -q   # 全绿
```

## 1. 冒烟(CPU 1 步 + MPS 50 步,已执行 ✓)

- CPU:`--model-size mid --steps 1 ...` → `params=37,675,042`
- MPS 50 步(batch 8 / seq 384):loss 10.9 → 5.3 正常下降,吞吐 **~3,000 tok/s**

## 2. 全量 v9 从头训练(30k steps ≈ 8.5h,过夜)

```bash
nohup env HF_HUB_DISABLE_XET=1 .venv-arm64/bin/python -u -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --steps 30000 --batch-size 8 --seq-len 384 --device mps \
  --lr 3e-4 --eos-loss-weight 4.0 \
  --model-size mid \
  --train-tokens 20000000 --eval-tokens 200000 \
  --eval-every 1000 --eval-batches 20 --log-every 100 --save-every 1000 \
  --seed 7 --out checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  > runs/train_v9_mid_scaling.log 2>&1 &
```

- 监控:`tail -f runs/train_v9_mid_scaling.log` / `grep "val loss" runs/train_v9_mid_scaling.log`
- 中断续跑:同命令改 `--resume checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt`(去掉隐式 from-scratch 语义)
- 到 30k 后如需继续冲验收:同命令 `--resume` + `--steps` 加大增量
- 如 MPS OOM:`--batch-size 4`(吞吐相应减半)
- **数据预算红线(T3b 轮教训):20M tokens ≈ 9 万窗口,30k 步 × batch 8 ≈ 1.6 epochs,安全区;不得下调 `--train-tokens`**

## 3. 验收(对照迭代计划 T3b)

| 指标 | v6.1(small)基线 | v9(mid)目标 |
|---|---|---|
| val loss | 1.6227 | **< 1.55** |
| story endings | 33/60 | **≥ 40/60** |
| EOS | 60/60 | **60/60 红线** |
| weak_prompt_adherence | 29 | 不回退 |

```bash
.venv-arm64/bin/python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  --device mps --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --preset-set coherence-v2 --max-new-tokens 256 --stop-at-eos \
  --require-eos 60/60 --fail-on weak_prompt_adherence=29 \
  --fail-on missing_ending=27 --fail-on repeated_trigram=14 --out-dir runs
```

- 注:coherence-v2 preset 参数是按 small 模型调的;mid 首评可另跑一版 `--max-new-tokens 376`(全 seq384 预算)对照
- 趋势对比:`examples.compare_small_lm_eval_reports --glob 'runs/generation_eval_*.md' --latest 3`

## 4. 里程碑参考(small 谱系,供中途判读)

| steps | small val | mid 预期 |
|---|---|---|
| 5k | 2.56 | 显著更低(容量×4) |
| 20k | 2.16 | ~1.9 或更低 |
| 30k | 1.91 | 待实测 |
| 105k | 1.62 | 验收目标 < 1.55 需续跑至更多步 |

## 5. 归档 checklist

- [x] checkpoint 复制 `checkpoints/openmythos_tinystories_mid_38m_step60000_seq384_v9.pt` + SHA256(`43a88b98…`)
- [x] 模型卡 `docs/model_cards/openmythos_tinystories_mid_38m_step60000_seq384_v9.md`
- [x] AGENTS.md 执行日志 + 滚动快照更新

## 6. 执行结果(2026-10-24 · 60k 完成 + mid-v1 preset)

| 阶段 | val | EOS(256/376) | endings | repeated | 备注 |
|---|---|---|---|---|---|
| 30k | 1.5515 | 58/60 → 60/60 | 25-26/60 | 17-18 | EOS 有预算长尾 |
| 45k | 1.5142 ✓ | **60/60 双预算** | 30/60 | 14 | 生成素养追平期 |
| 60k(coherence-v2) | 1.5130 ✓ | 58-59/60 ❌ | 30-31/60(平台) | **22-23 ❌** | ~9.2 epochs,记忆化前兆 |
| 60k(**mid-v1**) | 同 | **60/60 严格-256** | **35/60** | 17 | 推荐配置 |

**判决:** val ✓ · EOS 红线 ✓(需 mid-v1 preset)· endings 35/60 vs ≥40 ❌。45k→60k 生成指标平台化且
repeated 恶化 → **停止堆步数**;endings/weak 的下一杠杆在数据侧(20M 预算 9.2 epochs 是结构性原因)。

**教训(已入日志):** `--resume` 续跑前必须先归档当前 checkpoint——45k 的最佳生成权重被 60k 覆盖且不可恢复。

## 7. T5 · 新鲜数据续跑(2026-10-24 启动)

T3b 终判后 weak(35 vs 29)/repeated(17 vs 14)的结构性根因是 20M 预算 × 9.2 epochs。
T5 以 **60M 预算(3 倍)resume 续跑至 75k**:数据集按流序确定性收集,旧 52k 窗口之外新增
~104k 新鲜窗口(67% 未见过),15k 步 ≈ 0.77 epochs,远离记忆化区间。

```bash
nohup env HF_HUB_DISABLE_XET=1 .venv-arm64/bin/python -u -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --resume checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  --steps 15000 --batch-size 8 --seq-len 384 --device mps \
  --lr 3e-4 --eos-loss-weight 4.0 --model-size mid \
  --train-tokens 60000000 --eval-tokens 200000 \
  --eval-every 1000 --eval-batches 20 --log-every 100 --save-every 1000 \
  --seed 7 --out checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  > runs/train_v9_t5_freshdata.log 2>&1 &
```

- **续跑前归档检查**:60k 已归档(`openmythos_tinystories_mid_38m_step60000_seq384_v9.pt`)✓
- 验收(mid-v1 preset,256 严格):endings ≥ 40/60 · weak ≤ 29 · repeated ≤ 14 · EOS 60/60 · val 回退 ≤ 0.01
- 判读:endings 35→38+ 则继续加数据量;平台化则 T5 判负,结论收敛为"TinyStories 语料质量上限"
