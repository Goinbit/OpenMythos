"""CPU timings for the paths Stage 3 was supposed to speed up.

Stage 3 (P3.1 SDPA, P3.2 vectorized MoE dispatch, P3.5 no per-loop early-exit
sync) shipped with equivalence tests but no benchmark numbers: the acceptance
criteria were deferred to "CI/GPU" and nothing ever measured them. Without a
harness, "no unmeasured performance change" stays a rule with no enforcement,
and the next performance PR still has nothing to diff against.

Two case kinds, deliberately different:

* ratio cases (``sdpa_attention``, ``vectorized_moe``) run the pre-optimization
  reference and the shipped path on identical inputs and print base/fast. The
  reference is checked for numerical equivalence *before* timing, because a
  speedup measured across two different functions is not a speedup.
* absolute cases (``prefill``, ``backward``, ``decode_step``) have no
  pre-optimization twin, so they print milliseconds and become the before/after
  anchor for future PRs.

Numbers are single-process CPU wall-clock. They bound CPU-side cost, not the
GPU throughput the Stage 3 exit gate asks for — those still need hardware.
Thread count is pinned so runs stay comparable, and the payload records torch
version, threads and the CPU the kernels ran on: the first two-platform run
inverted both ratios at identical config (sdpa 1.07x on arm64 vs 1.55x on the
x86 runner, moe 3.39x vs 2.73x). A timing without those coordinates is not
comparable to anything, including another timing.

Run:
    python -m benchmarks.gate_bench
    python -m benchmarks.gate_bench --seq-len 1024 --repeat 9
    python -m benchmarks.gate_bench --json /tmp/gate_bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from open_mythos.main import (
    GQAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    apply_rope,
    precompute_rope_freqs,
)

EQUIV_TOL = 1e-4
DECODE_TOKENS = 4
# The decode case needs RoPE/cache room past the timed sequence, so
# max_seq_len carries slack and the timed length is derived back from it.
MAX_SEQ_SLACK = 32
# The pre-P3.2 reference is a python loop per token-slot, so it is timed on a
# short sequence; the ratio it yields is about dispatch, not token count.
MOE_REF_SEQ_LEN = 64

_CFG_DEFAULTS = dict(
    vocab_size=1024,
    dim=512,
    n_heads=8,
    n_kv_heads=2,
    expert_dim=1024,
    n_experts=64,
    n_experts_per_tok=8,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    act_threshold=0.99,
    lora_rank=32,
    dropout=0.0,
)


@dataclass(frozen=True)
class Case:
    """One measured line: a base/fast pair, or a single absolute timing."""

    name: str
    base_ms: Optional[float]
    fast_ms: float
    max_diff: Optional[float]

    @property
    def ratio(self) -> Optional[float]:
        if self.base_ms is None or self.base_ms <= 0 or self.fast_ms <= 0:
            return None
        return self.base_ms / self.fast_ms


def make_cfg(*, seq_len: int, n_experts: int) -> MythosConfig:
    """Config for a timing run: shape knobs from the CLI, the rest fixed."""
    return MythosConfig(
        **{
            **_CFG_DEFAULTS,
            "n_experts": n_experts,
            "n_experts_per_tok": min(_CFG_DEFAULTS["n_experts_per_tok"], n_experts),
            "max_seq_len": seq_len + MAX_SEQ_SLACK,
        }
    )


def timed_seq_len(cfg: MythosConfig) -> int:
    return cfg.max_seq_len - MAX_SEQ_SLACK


def timed_ms(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    """Median wall-clock of ``fn`` over ``repeat`` samples, after warmup."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def require_equivalent(base: torch.Tensor, fast: torch.Tensor, *, case: str) -> float:
    """Return max abs difference, raising if the two paths are not the same."""
    if base.shape != fast.shape:
        raise RuntimeError(
            f"{case}: shape mismatch — reference produced {tuple(base.shape)}, "
            f"shipped path produced {tuple(fast.shape)}; "
            "timing them together is meaningless"
        )
    diff = (base - fast).abs().max().item()
    if not math.isfinite(diff):
        raise RuntimeError(f"{case}: difference is not finite ({diff})")
    if diff > EQUIV_TOL:
        raise RuntimeError(
            f"{case}: reference and shipped path differ by {diff:.3e} "
            f"(tolerance {EQUIV_TOL:.1e}); fix equivalence before timing"
        )
    return diff


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
    return mask.triu(diagonal=1)


def _manual_attention(
    attn: GQAttention, x: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Pre-P3.1 path: explicit QK^T/sqrt(d) + causal mask -> softmax -> @V."""
    b, t, _ = x.shape
    q = attn.wq(x).view(b, t, attn.n_heads, attn.head_dim)
    k = attn.wk(x).view(b, t, attn.n_kv_heads, attn.head_dim)
    v = attn.wv(x).view(b, t, attn.n_kv_heads, attn.head_dim)
    q = apply_rope(q, freqs)
    k = apply_rope(k, freqs)
    k = k.repeat_interleave(attn.groups, dim=2)
    v = v.repeat_interleave(attn.groups, dim=2)
    q, k, v = (m.transpose(1, 2) for m in (q, k, v))
    scores = q @ k.transpose(-2, -1) * attn.head_dim**-0.5
    probs = torch.softmax(scores + _causal_mask(t, x.device), dim=-1)
    out = (probs @ v).transpose(1, 2).contiguous().view(b, t, -1)
    return attn.wo(out)


def bench_attention(cfg: MythosConfig, *, warmup: int, repeat: int) -> Case:
    torch.manual_seed(0)
    attn = GQAttention(cfg).eval()
    seq_len = timed_seq_len(cfg)
    x = torch.randn(1, seq_len, cfg.dim)
    freqs = precompute_rope_freqs(attn.head_dim, cfg.max_seq_len, cfg.rope_theta)[
        :seq_len
    ]
    mask = _causal_mask(seq_len, x.device)

    ref = _manual_attention(attn, x, freqs)
    got = attn(x, freqs, mask=mask)
    max_diff = require_equivalent(ref, got, case="sdpa_attention")

    base_ms = timed_ms(
        lambda: _manual_attention(attn, x, freqs), warmup=warmup, repeat=repeat
    )
    fast_ms = timed_ms(
        lambda: attn(x, freqs, mask=mask), warmup=warmup, repeat=repeat
    )
    return Case("sdpa_attention", base_ms, fast_ms, max_diff)


def _loop_moe(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """Pre-P3.2 path: python loop over tokens and expert slots."""
    b, t, d = x.shape
    flat = x.reshape(b * t, d)
    logits = moe.router.linear(flat)
    scores = torch.softmax(logits, dim=-1)
    _, topk_idx = (logits + moe.router.router_bias).topk(moe.topk, dim=-1)
    topk_scores = scores.gather(-1, topk_idx)
    topk_scores = topk_scores / topk_scores.sum(-1, keepdim=True).clamp(min=1e-9)
    experts = moe.routed_experts
    gate_weight = experts._effective_weight(experts.gate_weight)
    up_weight = experts._effective_weight(experts.up_weight)
    down_weight = experts._effective_weight(experts.down_weight)
    out = torch.zeros_like(flat)
    for n in range(flat.shape[0]):
        for slot in range(moe.topk):
            eid = topk_idx[n, slot].item()
            hidden = F.silu(F.linear(flat[n : n + 1], gate_weight[eid])) * F.linear(
                flat[n : n + 1], up_weight[eid]
            )
            expert_out = F.linear(hidden, down_weight[eid]).squeeze(0)
            out[n] = out[n] + topk_scores[n, slot] * expert_out
    for shared in moe.shared_experts:
        out = out + shared(flat)
    return out.view(b, t, d)


def bench_moe(cfg: MythosConfig, *, warmup: int, repeat: int) -> Case:
    torch.manual_seed(0)
    moe = MoEFFN(cfg).eval()
    x = torch.randn(1, min(timed_seq_len(cfg), MOE_REF_SEQ_LEN), cfg.dim)

    ref = _loop_moe(moe, x)
    got = moe(x)
    max_diff = require_equivalent(ref, got, case="vectorized_moe")

    base_ms = timed_ms(lambda: _loop_moe(moe, x), warmup=warmup, repeat=repeat)
    fast_ms = timed_ms(lambda: moe(x), warmup=warmup, repeat=repeat)
    return Case("vectorized_moe", base_ms, fast_ms, max_diff)


def bench_model(cfg: MythosConfig, *, warmup: int, repeat: int) -> list[Case]:
    """Absolute timings: no pre-optimization twin exists for these."""
    torch.manual_seed(0)
    model = OpenMythos(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, timed_seq_len(cfg)))

    def prefill() -> None:
        with torch.no_grad():
            model(ids)

    def backward() -> None:
        model.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        model.zero_grad(set_to_none=True)

    def decode() -> None:
        model.generate(ids, max_new_tokens=DECODE_TOKENS, n_loops=cfg.max_loop_iters)

    return [
        Case("prefill", None, timed_ms(prefill, warmup=warmup, repeat=repeat), None),
        Case("backward", None, timed_ms(backward, warmup=warmup, repeat=repeat), None),
        Case(
            "decode_step",
            None,
            timed_ms(decode, warmup=warmup, repeat=repeat) / DECODE_TOKENS,
            None,
        ),
    ]


def render_lines(cases: list[Case]) -> list[str]:
    lines = []
    for case in cases:
        if case.ratio is None:
            lines.append(f"[gate-bench] case={case.name} value_ms={case.fast_ms:.3f}")
        else:
            lines.append(
                f"[gate-bench] case={case.name} base_ms={case.base_ms:.3f}"
                f" fast_ms={case.fast_ms:.3f} ratio={case.ratio:.2f}"
                f" max_diff={case.max_diff:.2e}"
            )
    return lines


def to_payload(cases: list[Case], meta: dict[str, object]) -> dict[str, object]:
    return {
        "meta": meta,
        "cases": [
            {
                "name": case.name,
                "base_ms": case.base_ms,
                "fast_ms": case.fast_ms,
                "ratio": case.ratio,
                "max_diff": case.max_diff,
            }
            for case in cases
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads; pinned so runs stay comparable",
    )
    parser.add_argument("--json", type=str, default="", help="write payload here")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    cfg = make_cfg(seq_len=args.seq_len, n_experts=args.n_experts)
    cases: list[Case] = [
        bench_attention(cfg, warmup=args.warmup, repeat=args.repeat),
        bench_moe(cfg, warmup=args.warmup, repeat=args.repeat),
        *bench_model(cfg, warmup=args.warmup, repeat=args.repeat),
    ]
    for line in render_lines(cases):
        print(line, flush=True)

    if args.json:
        meta: dict[str, object] = {
            "torch": torch.__version__,
            "threads": args.threads,
            "cpu": f"{platform.machine()}/{torch.backends.cpu.get_cpu_capability()}",
            "seq_len": args.seq_len,
            "n_experts": args.n_experts,
            "dim": cfg.dim,
            "warmup": args.warmup,
            "repeat": args.repeat,
        }
        with open(args.json, "w") as handle:
            json.dump(to_payload(cases, meta), handle, indent=2)
        print(f"[gate-bench] json written to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
