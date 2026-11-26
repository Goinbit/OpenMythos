"""Executable evidence for ACT fixed-loop and host-sync behaviour."""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from open_mythos import MythosConfig, OpenMythos, RecurrentBlock
from open_mythos.primitives.rope import precompute_rope_freqs


def _cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=100,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )


def _timed_ms(fn, repeat: int) -> float:
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


class _SyncCounter(TorchDispatchMode):
    def __init__(self):
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.all.default,
        ):
            self.calls += 1
        return func(*args, **kwargs)


def bench_act_sync(n_loops: int = 8) -> dict:
    base_cfg = _cfg()
    cfg = dataclasses.replace(
        base_cfg,
        recurrence=dataclasses.replace(base_cfg.recurrence, max_loop_iters=n_loops),
    )
    block = RecurrentBlock(cfg).eval()
    freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
    h = torch.randn(1, 4, cfg.dim)
    e = torch.randn(1, 4, cfg.dim)
    with torch.no_grad():
        block.act.halt.weight.zero_()
        block.act.halt.bias.fill_(torch.logit(torch.tensor(0.999)).item())
    with _SyncCounter() as counter:
        block(h, e, freqs[:4], n_loops=n_loops)
    return {
        "loops": n_loops,
        "host_syncs": counter.calls,
        "allowed_syncs": n_loops // 4,
    }


def _legacy_act_forward(block, h, e, freqs, n_loops: int):
    halted = torch.zeros_like(h[..., 0], dtype=torch.bool)
    cumulative_p = torch.zeros_like(h[..., 0])
    h_out = torch.zeros_like(h)
    for loop_t in range(n_loops):
        running = ~halted
        h_loop = torch.where(
            running.unsqueeze(-1),
            h + block.loop_idx_table[loop_t].view(1, 1, -1),
            h,
        )
        trans_out = block.block(
            block.norm(h_loop + e), freqs, None, None, f"legacy_{loop_t}"
        )
        trans_out = trans_out + block.lora(trans_out, loop_t)
        trans_out = torch.where(
            running.unsqueeze(-1), trans_out, torch.zeros_like(trans_out)
        )
        h = torch.where(
            running.unsqueeze(-1), block.injection(h, e, trans_out), h
        )
        p = torch.where(running, block.act(h), torch.zeros_like(h[..., 0]))
        remainder = (1.0 - cumulative_p).clamp(min=0)
        weight = torch.where(
            (cumulative_p + p >= block.cfg.act_threshold)
            | (loop_t == n_loops - 1),
            remainder,
            p,
        ) * running.float()
        h_out = h_out + weight.unsqueeze(-1) * h
        cumulative_p = cumulative_p + p * running.float()
        halted = halted | (cumulative_p >= block.cfg.act_threshold)
        if halted.all():
            break
    return h_out


@torch.no_grad()
def bench_act_timing(n_loops: int = 8, repeat: int = 3) -> dict:
    base_cfg = _cfg()
    cfg = dataclasses.replace(
        base_cfg,
        recurrence=dataclasses.replace(base_cfg.recurrence, max_loop_iters=n_loops),
    )
    block = RecurrentBlock(cfg).eval()
    freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
    h = torch.randn(1, 4, cfg.dim)
    e = torch.randn(1, 4, cfg.dim)
    block.act.halt.weight.zero_()
    block.act.halt.bias.fill_(-10.0)

    def reference():
        return _legacy_act_forward(block, h, e, freqs[:4], n_loops)

    def fixed():
        return block(h, e, freqs[:4], n_loops=n_loops)

    max_diff = float((reference() - fixed()).abs().max())
    base_ms = _timed_ms(reference, repeat)
    fast_ms = _timed_ms(fixed, repeat)
    return {
        "loops": n_loops,
        "base_ms": base_ms,
        "fast_ms": fast_ms,
        "ratio": base_ms / fast_ms,
        "max_diff": max_diff,
    }


@torch.no_grad()
def bench_generate_sync(n_loops: int = 8, max_new_tokens: int = 2) -> dict:
    base_cfg = _cfg()
    cfg = dataclasses.replace(
        base_cfg,
        recurrence=dataclasses.replace(base_cfg.recurrence, max_loop_iters=n_loops),
    )
    model = OpenMythos(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    with _SyncCounter() as counter:
        model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            temperature=1e-8,
            top_k=0,
        )
    return {
        "loops": n_loops,
        "new_tokens": max_new_tokens,
        "host_syncs": counter.calls,
        "allowed_syncs": n_loops // 4,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    payload = {
        "meta": {
            "torch": torch.__version__,
            "threads": args.threads,
            "cpu": f"{platform.machine()}/{torch.backends.cpu.get_cpu_capability()}",
            "repeat": args.repeat,
        },
        "act_sync": bench_act_sync(),
        "generate_sync": bench_generate_sync(),
        "act_timing": bench_act_timing(repeat=args.repeat),
    }
    print("[recurrent-bench] " + json.dumps(payload), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
