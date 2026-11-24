"""Tests for the Stage 3 benchmark harness.

The harness prints two kinds of claim: a speedup ratio and an absolute
timing. A ratio is only meaningful when the reference path and the shipped
path compute the same function, so the equivalence guard — not the stopwatch
— is the part these tests pin down hardest.
"""

from __future__ import annotations

import json

import pytest
import torch

from benchmarks import gate_bench
from benchmarks.gate_bench import (
    Case,
    bench_attention,
    bench_model,
    bench_moe,
    main,
    make_cfg,
    render_lines,
    require_equivalent,
    timed_ms,
    to_payload,
)


class TestTimedMs:
    def test_reports_median_of_repeat_samples(self, monkeypatch):
        ticks = iter([0.0, 10.0, 0.0, 30.0, 0.0, 20.0, 0.0, 40.0])
        monkeypatch.setattr(gate_bench.time, "perf_counter", lambda: next(ticks))
        assert timed_ms(lambda: None, warmup=1, repeat=4) == pytest.approx(25_000.0)

    def test_warms_up_before_sampling(self, monkeypatch):
        monkeypatch.setattr(gate_bench.time, "perf_counter", lambda: 0.0)
        calls: list[int] = []
        timed_ms(lambda: calls.append(1), warmup=2, repeat=3)
        assert len(calls) == 5

    def test_rejects_zero_repeat(self):
        with pytest.raises(ValueError):
            timed_ms(lambda: None, warmup=0, repeat=0)


class TestCase:
    def test_ratio_is_base_over_fast(self):
        assert Case("sdpa_attention", 12.0, 3.0, 1e-6).ratio == pytest.approx(4.0)

    def test_absolute_case_has_no_ratio(self):
        assert Case("prefill", None, 3.0, None).ratio is None


class TestRequireEquivalent:
    def test_returns_max_abs_difference(self):
        base = torch.zeros(2, 3)
        fast = base.clone()
        fast[1, 2] = 1e-6
        assert require_equivalent(base, fast, case="demo") == pytest.approx(1e-6)

    def test_rejects_value_drift(self):
        with pytest.raises(RuntimeError, match="demo"):
            require_equivalent(torch.zeros(4), torch.ones(4), case="demo")

    def test_rejects_shape_mismatch(self):
        with pytest.raises(RuntimeError, match="shape"):
            require_equivalent(torch.zeros(4), torch.zeros(5), case="demo")

    def test_rejects_non_finite_difference(self):
        nan = torch.full((4,), float("nan"))
        with pytest.raises(RuntimeError, match="finite"):
            require_equivalent(torch.zeros(4), nan, case="demo")


class TestRatioCases:
    def test_attention_reference_matches_shipped_path(self):
        cfg = make_cfg(seq_len=16, n_experts=4)
        case = bench_attention(cfg, warmup=1, repeat=2)
        assert case.name == "sdpa_attention"
        assert case.max_diff is not None
        assert case.max_diff < 1e-4
        assert case.base_ms is not None and case.base_ms > 0
        assert case.fast_ms > 0

    def test_moe_reference_matches_shipped_path(self):
        cfg = make_cfg(seq_len=12, n_experts=8)
        case = bench_moe(cfg, warmup=1, repeat=2)
        assert case.name == "vectorized_moe"
        assert case.max_diff is not None
        assert case.max_diff < 1e-4
        assert case.ratio is not None


class TestEquivalenceGateIsLoadBearing:
    def test_bench_attention_propagates_equivalence_failure(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic drift")

        monkeypatch.setattr(gate_bench, "require_equivalent", boom)
        cfg = make_cfg(seq_len=8, n_experts=4)
        with pytest.raises(RuntimeError, match="synthetic drift"):
            bench_attention(cfg, warmup=1, repeat=1)

    def test_bench_moe_propagates_equivalence_failure(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic drift")

        monkeypatch.setattr(gate_bench, "require_equivalent", boom)
        cfg = make_cfg(seq_len=8, n_experts=4)
        with pytest.raises(RuntimeError, match="synthetic drift"):
            bench_moe(cfg, warmup=1, repeat=1)


class TestAbsoluteCases:
    def test_model_cases_are_absolute_and_positive(self):
        cfg = make_cfg(seq_len=8, n_experts=4)
        cases = bench_model(cfg, warmup=1, repeat=1)
        assert [case.name for case in cases] == ["prefill", "backward", "decode_step"]
        for case in cases:
            assert case.base_ms is None
            assert case.fast_ms > 0
            assert case.max_diff is None


class TestRender:
    def test_ratio_case_renders_base_fast_and_diff(self):
        (line,) = render_lines([Case("vectorized_moe", 20.0, 4.0, 1.5e-6)])
        assert line.startswith("[gate-bench] case=vectorized_moe")
        assert "base_ms=20.000" in line
        assert "fast_ms=4.000" in line
        assert "ratio=5.00" in line
        assert "max_diff=1.50e-06" in line

    def test_absolute_case_renders_single_value(self):
        (line,) = render_lines([Case("prefill", None, 7.25, None)])
        assert line == "[gate-bench] case=prefill value_ms=7.250"
        assert "ratio=" not in line


class TestPayload:
    def test_payload_carries_meta_and_case_fields(self):
        cases = [Case("sdpa_attention", 4.0, 1.0, 1e-6), Case("prefill", None, 2.0, None)]
        payload = to_payload(cases, {"torch": "2.12.0", "threads": 1})
        assert payload["meta"] == {"torch": "2.12.0", "threads": 1}
        assert [case["name"] for case in payload["cases"]] == [
            "sdpa_attention",
            "prefill",
        ]
        assert payload["cases"][0]["ratio"] == pytest.approx(4.0)
        assert payload["cases"][1]["ratio"] is None


class TestMain:
    def test_smoke_run_prints_and_writes_json(self, tmp_path, capsys):
        out = tmp_path / "bench.json"
        rc = main(
            [
                "--seq-len",
                "8",
                "--n-experts",
                "4",
                "--warmup",
                "1",
                "--repeat",
                "1",
                "--json",
                str(out),
            ]
        )
        assert rc == 0
        printed = capsys.readouterr().out
        assert "[gate-bench] case=sdpa_attention" in printed
        assert "[gate-bench] case=prefill" in printed
        payload = json.loads(out.read_text())
        assert {case["name"] for case in payload["cases"]} == {
            "sdpa_attention",
            "vectorized_moe",
            "prefill",
            "backward",
            "decode_step",
        }

    def test_meta_records_the_stack_behind_the_numbers(self, tmp_path, capsys):
        out = tmp_path / "bench.json"
        main(
            [
                "--seq-len",
                "8",
                "--n-experts",
                "4",
                "--warmup",
                "1",
                "--repeat",
                "1",
                "--threads",
                "1",
                "--json",
                str(out),
            ]
        )
        capsys.readouterr()
        meta = json.loads(out.read_text())["meta"]
        assert meta["torch"] == torch.__version__
        assert meta["threads"] == 1
        assert meta["seq_len"] == 8
