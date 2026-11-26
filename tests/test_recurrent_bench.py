"""Executable evidence for P3.5 ACT fixed-loop and host-sync behaviour."""
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


def test_main_writes_payload(tmp_path, capsys):
    output = tmp_path / "recurrent.json"
    rc = recurrent_bench.main(["--repeat", "1", "--json", str(output)])
    assert rc == 0
    assert "[recurrent-bench]" in capsys.readouterr().out
    payload = json.loads(output.read_text())
    assert set(payload) == {"meta", "act_sync", "generate_sync", "act_timing"}
    assert payload["meta"]["repeat"] == 1
