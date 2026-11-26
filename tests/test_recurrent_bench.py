"""Executable evidence for ACT, loop embedding, and causal-mask behaviour."""
from __future__ import annotations

import json

from benchmarks import recurrent_bench


def test_act_forward_host_sync_count():
    result = recurrent_bench.bench_act_sync(n_loops=8)
    assert result["host_syncs"] == 0
    assert result["allowed_syncs"] == 2


def test_generate_host_sync_count():
    result = recurrent_bench.bench_generate_sync(n_loops=8, max_new_tokens=2)
    assert result["host_syncs"] == 0
    assert result["allowed_syncs"] == 2


def test_act_timing_matches_reference():
    result = recurrent_bench.bench_act_timing(n_loops=4, repeat=2)
    assert result["max_diff"] < 1e-6
    assert result["ratio"] > 0


def test_loop_embedding_matches_and_is_faster():
    result = recurrent_bench.bench_loop_embedding(repeat=3)
    assert result["max_diff"] < 1e-6
    assert result["ratio"] > 1.0


def test_causal_mask_free_forward_matches_reference():
    result = recurrent_bench.bench_causal_mask(seq_len=32, repeat=1)
    assert result["max_diff"] < 1e-6
    assert result["base_mask_ops"] == 2
    assert result["fast_mask_ops"] == 0
    assert result["base_mask_elements"] == 2 * 32 * 32
    assert result["fast_mask_elements"] == 0
    assert result["ratio"] > 0


def test_main_writes_payload(tmp_path, capsys):
    output = tmp_path / "recurrent.json"
    rc = recurrent_bench.main(["--repeat", "1", "--json", str(output)])
    assert rc == 0
    assert "[recurrent-bench]" in capsys.readouterr().out
    payload = json.loads(output.read_text())
    assert set(payload) == {
        "meta",
        "act_sync",
        "generate_sync",
        "act_timing",
        "loop_embedding",
        "causal_mask",
    }
    assert payload["meta"]["repeat"] == 1
