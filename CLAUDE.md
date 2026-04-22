# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

# OpenMythos — 优化方案与验收标准

**Project context:** OpenMythos v0.4.0 — PyTorch Recurrent-Depth Transformer (Prelude → Recurrent Block → Coda), MLA/GQA attention, MoE FFN, ACT halting, LoRA depth adapter, LTI injection.

**Companion documents:**
- [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) — full multi-agent analysis
- [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) — detailed plan (source of truth)

All work in this project must reference the phase/item below when opening a PR (e.g. `[P0.3]`, `[P3.1]`).

---

## 阶段 0 — 紧急修复

**目标:** 消除静默失败与导入错误,1 天内完成。所有后续工作的前置依赖。

| ID | 任务 | 位置 | 验收标准 |
|---|---|---|---|
| **P0.1** | 修复 `__init__.py` 幽灵导出 | `open_mythos/__init__.py:52-53` | ✅ `python -c "from open_mythos import *"` 无 `AttributeError`<br>✅ 新增测试:`test_public_api_importable` |
| **P0.2** | 训练环 NaN/Inf 守护 | `training/3b_fine_web_edu.py:499-506` | ✅ 非有限 loss 立刻 `RuntimeError`<br>✅ checkpoint 保存前扫描所有参数 `isfinite`<br>✅ 注入 NaN 的单元测试验证阻断 |
| **P0.3** | MoE topk 除零 clamp | `open_mythos/main.py:488` | ✅ `clamp(min=1e-9)` 在归一化前应用<br>✅ 单元测试:全 0 topk_scores 不产生 NaN |
| **P0.4** | `attn_type` Literal + 校验 | `main.py:53, 621, 983` | ✅ 类型标注 `Literal["gqa","mla"]`<br>✅ 错拼 `"MLA"`/`"gqa "` 在 `MythosConfig` 构造时 `ValueError`<br>✅ 单元测试覆盖错拼情形 |
| **P0.5** | 训练复现性 seed | `training/3b_fine_web_edu.py:main()` | ✅ 同 seed 两次运行前 10 步 loss 完全一致<br>✅ DataLoader worker_init_fn 设 seed<br>✅ `torch.backends.cudnn.deterministic=True` 可选切换 |

**阶段 0 退出门槛(全部满足):**
- `pytest tests/` 全绿
- `python -c "from open_mythos import *"` 无报错
- 人为注入 NaN / 错拼 attn_type 均被立刻拦截

---

## 阶段 1 — 契约硬化

**目标:** 所有配置不变量在构造时立刻失败,消除静默配置错误。

| ID | 任务 | 位置 | 验收标准 |
|---|---|---|---|
| **P1.1** | `MythosConfig` 冻结 + `__post_init__` | `main.py:9-74` | ✅ `frozen=True`<br>✅ 校验:`dim%n_heads==0`、`n_heads%n_kv_heads==0`(GQA)、`n_experts_per_tok≤n_experts`、`act_threshold∈(0,1]`、`lora_rank≤dim`<br>✅ 每条不变量都有单元测试故意违反 → 期望 `AssertionError`/`ValueError`<br>✅ 训练脚本改用 `dataclasses.replace` 而非直接 mutation |
| **P1.2** | ACT halted 位置跳过计算 | `main.py:829-861` | ✅ 强制 iter 1 全部 halt 场景,`h_out ≈ h_after_iter1`(数值容差 1e-5)<br>✅ halted 位置的 hidden state `h` 不再被 LTI 更新(或等价的门控)<br>✅ `test_act_halted_hidden_state_frozen` 通过 |
| **P1.3** | LoRA extrapolation warning | `main.py:588` | ✅ `loop_t > max_t` 时 `warnings.warn` 一次<br>✅ 测试:`pytest.warns` 捕获该 warning |
| **P1.4** | Tokenizer vs Config vocab_size 校验 | `training/3b_fine_web_edu.py:365-399` | ✅ 不匹配时启动即 `ValueError`(错误消息包含两个值)<br>✅ `load_checkpoint` 读 `ckpt["vocab_size"]` 与当前比较<br>✅ 单元测试:vocab_size 漂移的 checkpoint 被拒绝 |
| **P1.5** | `generate()` max_seq_len 越界 | `main.py:1031-1047` | ✅ `start_pos + 1 > max_seq_len` 时 break(或 raise,据语义决定)<br>✅ 单元测试 `generate` 接近 `max_seq_len` 不崩溃也不静默越界 |

**阶段 1 退出门槛:**
- 所有 `variants.py` 预设构造时通过 `__post_init__`
- 人为构造违反不变量 → 构造时失败(非首次 forward 时失败)
- `mypy --strict open_mythos/` 对 `attn_type` 路径无警告

---

## 阶段 2 — 回归测试补齐

**目标:** 最近 5 个修复的 bug 全部拥有专属回归,测试覆盖从 ~40% 提至 ≥70%。

| ID | 任务 | 测试文件 | 验收标准 |
|---|---|---|---|
| **P2.1** | ACT 权重守恒 + 强制 halt | `test_main.py::TestACT` | ✅ `test_act_weights_sum_to_one`:Σ weight ≈ 1 per position<br>✅ `test_act_forced_halt_matches_hidden` |
| **P2.2** | KV cache loop-key 完整性 | `test_main.py::TestKVCache` | ✅ `test_kv_cache_populates_all_loop_keys`<br>✅ 多步 decode 下 cache 每层每 loop 都写入 |
| **P2.3** | LoRA depth extrapolation clamp | `test_main.py::TestLoRAAdapter` | ✅ `test_lora_clamps_beyond_max_loops`<br>✅ `loop_t ≥ max_loops` 不 crash 且输出等于 `loop_t=max_loops-1` 的输出 |
| **P2.4** | RoPE decode 位置推进 | `test_main.py::TestRoPEExtended` | ✅ `test_rope_generate_uses_increasing_start_pos`<br>✅ 连续 2 步 decode 的 freqs_cis slice 不相同 |
| **P2.5** | `moda.py` 烟雾测试 | `tests/test_moda.py` (新) | ✅ MoDAModel forward shape<br>✅ balance loss 为标量且 > 0<br>✅ 完整路径梯度流通 |
| **P2.6** | `variants.py` 全覆盖 | `tests/test_variants.py` (新) | ✅ 所有 7 个 preset 可构造(`__post_init__` 通过)<br>✅ 1b preset 可完成 tiny forward pass(dim 缩小版) |
| **P2.7** | 训练端到端 | `tests/test_training.py` (新) | ✅ `test_overfit_single_batch`:10 步 AdamW,loss 单调下降<br>✅ `test_checkpoint_roundtrip`:save→load→参数完全相等 |
| **P2.8** | `test_rope_debug.py` 转为 pytest | `tests/test_rope_debug.py` | ✅ 所有 `print` 替换为 `assert` / `torch.testing.assert_close`<br>✅ `pytest tests/test_rope_debug.py` 能采集并运行全部 |

**阶段 2 退出门槛:**
- `pytest --cov=open_mythos --cov-report=term-missing` 覆盖率 ≥ 70%
- 5 个历史 bug 每个有至少 1 个专属回归测试,revert 修复 commit 则该测试必须失败
- CI 跑 `pytest -x` 绿色

---

## 阶段 3 — 性能融合

**目标:** 训练吞吐 ×2~3,峰值显存 −40%。每项改动必须用 benchmark 数据 gate。

| ID | 任务 | 位置 | 验收标准 |
|---|---|---|---|
| **P3.1** | SDPA / FlashAttention 注意力 | `main.py:242, 384` | ✅ `F.scaled_dot_product_attention` 替换手写 matmul<br>✅ MLA/GQA 与原实现在 fp32 下 `torch.allclose(rtol=1e-5)` 等价(小规模 shape)<br>✅ benchmark:seq_len=2048, bsz=4, head=16 下 ≥1.5× 加速<br>✅ 峰值显存降低 ≥30% |
| **P3.2** | 向量化 MoE 分派 | `main.py:469-505` | ✅ 移除所有 Python `for eid in range(n_experts)` 循环<br>✅ 移除 `.any()` / `.item()` CUDA sync<br>✅ 与旧实现前向数值等价(`torch.allclose`)<br>✅ 反向梯度等价(`.grad.allclose`)<br>✅ benchmark:n_experts=64 下 ≥3× MoE forward 加速 |
| **P3.3** | 静态预分配 KV cache | `main.py:229, 365` | ✅ cache 从 dict-of-growing 改为预分配 buffer<br>✅ 生成质量测试(greedy 解码相同 prompt 输出一致)<br>✅ 长 decode(512 tokens)内存分配数降低 ≥90% |
| **P3.4** | RoPE 去除 `.float()` | `main.py:145-165` | ✅ bf16 autocast 下 `apply_rope` 输入输出均为 bf16<br>✅ 数值等价于旧实现(小规模 `allclose(atol=1e-3)` for bf16) |
| **P3.5** | 消除 ACT `.all()` sync | `main.py:860` | ✅ 训练路径无 early-exit(固定 max_loop_iters)<br>✅ 生成路径 sync 次数 ≤ `max_loop_iters/K` (K=4)<br>✅ benchmark:单步前向时间 −15% 以上 |
| **P3.6** | LoRA scale 无标量 tensor 构造 | `main.py:589` | ✅ `forward` 无 `torch.tensor(t_idx, ...)`<br>✅ `torch.compile(model)` 不报 graph break 指向 LoRA<br>✅ 数值等价 |
| **P3.7** | `loop_index_embedding` 预计算 | `main.py:522-548` | ✅ 预计算为 buffer `(max_loops, loop_dim)`<br>✅ forward 内无 `torch.cat` / `torch.arange` 分配<br>✅ 数值等价 |
| **P3.8** | 训练脚本小优化 | `training/3b_fine_web_edu.py` | ✅ `zero_grad(set_to_none=True)`<br>✅ `gradient_checkpointing` 可通过 cfg 开关启用<br>✅ buf 切片改用 deque 或偏移,分配数 O(1) |

**阶段 3 退出门槛:**
- 3B 模型单卡前向 ≥ 1.8× baseline 吞吐
- 3B 模型峰值显存 ≤ 60% baseline
- 所有阶段 2 测试仍然绿
- 输出数值回归(固定 seed 下 10 步训练 loss)在 fp32 下等价,bf16 下容差 < 0.1%

---

## 阶段 4 — 架构重构

**目标:** 代码模块化,消除 moda.py 重复,定义扩展点。纯结构改动,不改变行为。

| ID | 任务 | 验收标准 |
|---|---|---|
| **P4.1** | 拆分 `main.py` 为 ≤400 行模块 | ✅ 新目录结构:`config.py` / `primitives/` / `attention/` / `moe/` / `recurrence/` / `blocks.py` / `model.py` / `generate.py`<br>✅ 每个文件 ≤400 行<br>✅ `open_mythos.*` 公共导入路径不变(通过 `__init__.py` 重导出)<br>✅ 所有既有测试无修改即可运行通过 |
| **P4.2** | 消除 `main.py` vs `moda.py` 重复 | ✅ 共享的 `RMSNorm`/`Expert` 在 `primitives/` 只有一份<br>✅ `moda.py` 的 `DeepSeekMoE`(含 balance loss)反迁移为主 `MoEFFN`<br>✅ `moda.py` 移至 `open_mythos/experimental/moda.py` 并在 README 标注实验性<br>✅ `moda.py` 的所有测试仍绿 |
| **P4.3** | 定义扩展 Protocol | ✅ `AttentionProtocol` / `RouterProtocol` / `InjectionProtocol` 在对应包中<br>✅ `build_attention(cfg)` / `build_router(cfg)` 工厂替代 `if cfg.attn_type` 分支<br>✅ 用例文档:新增第三种 attention 无需修改 `TransformerBlock` |
| **P4.4** | Config 嵌套拆分 | ✅ `AttentionConfig` / `MoEConfig` / `RecurrenceConfig` 分离<br>✅ 顶层 `MythosConfig` 组合嵌套,`__post_init__` 校验跨组不变量<br>✅ 提供 `MythosConfig.from_flat(...)` 兼容旧 API;`variants.py` 过渡完成 |
| **P4.5** | 训练脚本模块化 | ✅ `training/` 拆成 `data.py` / `trainer.py` / `eval.py` / `schedule.py`<br>✅ 入口脚本 `run_3b.py` ≤ 60 行<br>✅ 添加 eval hook(至少 held-out loss)<br>✅ 同 seed 下训练 loss 曲线与拆分前完全一致 |

**阶段 4 退出门槛:**
- 所有阶段 0~3 测试无修改即绿
- `ruff check .` 无新增错误
- 3B 训练 100 步 loss 与重构前在 fp32 下 bit-exact

---

## 阶段 5 — 大规模训练能力(长期)

**目标:** 支持 `variants.py` 中 ≥50B preset 的实际训练。

| ID | 任务 | 验收标准 |
|---|---|---|
| **P5.1** | Tensor Parallelism | ✅ `ColumnParallelLinear` / `RowParallelLinear` 抽象接入<br>✅ 2 路 TP 下 3B 模型 loss 与单卡 FSDP 等价(fp32)<br>✅ 集成测试 pass |
| **P5.2** | Expert Parallelism | ✅ MoE experts 可跨 rank 分片,all_to_all 分派<br>✅ 专家数 256 下吞吐线性扩展到 ≥ 75% |
| **P5.3** | Activation Checkpointing | ✅ 可通过 cfg 开关启用,默认对 `RecurrentBlock`<br>✅ 3B + loops=16 峰值显存降低 ≥ 50%<br>✅ loss 数值等价(重计算不改变梯度) |
| **P5.4** | Paged KV Cache | ✅ 长 decode(8K tokens)不因 cat-growth 分配失败<br>✅ 与 vLLM PagedAttention 协议兼容或等价 |
| **P5.5** | 长上下文 | ✅ 移除 `_causal_mask` 显式 `(T,T)` 分配,改用 SDPA `is_causal`<br>✅ seq_len=32K 下单卡可训练<br>✅ `variants.py` 中不切实际的 `max_seq_len` 降级为现实值 |

**阶段 5 退出门槛:** 任一 preset(1B~50B)在目标硬件上端到端跑通 ≥ 1000 步训练且 loss 平稳下降。

---

## 通用工作规范(所有阶段适用)

### 提交规范
- 每个 PR 引用一个 ID(如 `[P0.3] fix MoE topk divide-by-zero`)
- 阶段 0~3 一个 ID 一个 PR,避免串联
- 阶段 4 可按子模块分 PR,但必须保持 CI 绿

### 测试要求(覆盖项目 Python 规则)
- 所有新代码必须先写测试(TDD)
- 覆盖率目标 ≥ 80%(common/testing.md)
- 数值回归测试优先于 shape-only 测试
- 修复 bug 必须有对应 `test_regression_<bug_id>` 命名的测试

### 代码审查要求
- 所有 PR 必须通过 `ruff check` + `black --check`
- 数值/算法改动(阶段 3)必须附带 benchmark 数据
- 类型标注新代码必须完整(python/coding-style.md)

### 禁止项
- ❌ 跳过 `__post_init__` 校验(直接 `object.__setattr__`)
- ❌ 使用 `assert` 做运行时输入校验(可能被 `python -O` 消除)
- ❌ `print()` in library code;训练脚本只用 `loguru`
- ❌ 静默 fallback(typo attn_type、除零、空 topk 等)
- ❌ 无 seed 的训练运行
- ❌ 未 gate 的性能改动(必须 benchmark 数据证明收益)

---

## 推荐执行顺序

```
阶段 0 (紧急修复)                         ──┐
                                            ├──>  解锁所有后续工作
阶段 2 (回归测试) ──── 并行 ─────────────── ┘

阶段 1 (契约硬化)  ──>  阶段 3 (性能融合)  ──>  阶段 4 (架构重构)  ──>  阶段 5 (大规模)
```

测试先行(阶段 2 与阶段 0 并行),确保后续所有重构都有 safety net。
