"""Gate the README against the code it documents.

The README is the only file most users read, and it had drifted: it called a
preset helper that does not exist, computed the spectral radius of a 1-D
diagonal with ``linalg.eigvals`` (which raises), and pointed the training
commands at the deprecated Stage 4.5 shim. These tests treat the README as a
contract so the drift fails in CI instead of in a user's terminal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import open_mythos
from open_mythos import variants as variants_module

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)
_RELATIVE_LINK = re.compile(r"\]\(([^)]+)\)")
_VARIANT_NAME = re.compile(r"\bmythos_[a-z0-9]+\b")
_IMPORT_FROM = re.compile(r"from open_mythos import \(([^)]*)\)", re.DOTALL)


def _text() -> str:
    return README.read_text()


def _python_blocks() -> list[str]:
    return _CODE_BLOCK.findall(_text())


def _block_containing(snippet: str) -> str:
    hits = [b for b in _python_blocks() if snippet in b]
    assert len(hits) == 1, f"expected exactly one python block with {snippet!r}, got {len(hits)}"
    return hits[0]


class TestLinks:
    def test_relative_links_resolve(self) -> None:
        targets = {
            t for t in _RELATIVE_LINK.findall(_text()) if "://" not in t and not t.startswith("#")
        }
        assert targets, "README has no relative links — the parser is broken"
        missing = sorted(t for t in targets if not (REPO_ROOT / t).exists())
        assert missing == []


class TestVariantNames:
    def test_every_variant_mentioned_exists(self) -> None:
        presets = {n for n in dir(variants_module) if n.startswith("mythos_")}
        assert sorted(set(_VARIANT_NAME.findall(_text())) - presets) == []

    def test_every_preset_is_documented(self) -> None:
        presets = {n for n in dir(variants_module) if n.startswith("mythos_")}
        assert sorted(presets - set(_VARIANT_NAME.findall(_text()))) == []

    def test_variant_table_matches_configs(self) -> None:
        rows = {}
        for line in _text().splitlines():
            if line.startswith("| `mythos_"):
                cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
                rows[cells[0]] = cells[1:]

        def compact(value: int) -> str:
            if value >= 1_000_000:
                return f"{value // 1_000_000}M"
            if value >= 1_024:
                return f"{value // 1_024}k"
            return str(value)

        presets = {n for n in dir(variants_module) if n.startswith("mythos_")}
        assert rows.keys() == presets
        for name in sorted(presets):
            cfg = getattr(variants_module, name)()
            expected = [
                str(cfg.dim),
                str(cfg.n_experts),
                str(cfg.expert_dim),
                str(cfg.max_loop_iters),
                compact(cfg.max_seq_len),
                compact(cfg.max_output_tokens),
            ]
            assert rows[name] == expected, name

    def test_imported_names_are_public(self) -> None:
        block = _block_containing("from open_mythos import (")
        imported = [
            n.strip()
            for chunk in _IMPORT_FROM.findall(block)
            for n in chunk.split(",")
            if n.strip() and not n.strip().startswith("#")
        ]
        assert imported, "README imports nothing from open_mythos — the parser is broken"
        missing = sorted(n for n in imported if not hasattr(open_mythos, n))
        assert missing == []


class TestSnippetsExecute:
    def test_usage_block_runs_end_to_end(self) -> None:
        source = _block_containing("Spectral radius")
        exec(compile(source, "README.md:usage", "exec"), {"__name__": "readme_usage"})


class TestPackageAndTrainingClaims:
    def test_badge_tracks_the_published_package_name(self) -> None:
        text = _text()
        assert "https://pypi.org/project/open-mythos/" in text
        assert "https://img.shields.io/pypi/v/open-mythos" in text

    def test_optimizer_matches_entry_point(self) -> None:
        source = (REPO_ROOT / "training" / "run_3b.py").read_text()
        assert "torch.optim.AdamW" in source
        assert "| Optimizer | AdamW |" in _text()

    def test_dataset_and_tokenizer_match_sources(self) -> None:
        data_source = (REPO_ROOT / "training" / "data.py").read_text()
        tokenizer_source = (REPO_ROOT / "open_mythos" / "tokenizer.py").read_text()
        assert '"HuggingFaceFW/fineweb-edu"' in data_source
        assert 'DEFAULT_MODEL_ID = "openai/gpt-oss-20b"' in tokenizer_source
        assert "`HuggingFaceFW/fineweb-edu`" in _text()
        assert "`openai/gpt-oss-20b`" in _text()

    def test_schedule_claim_matches_implementation(self) -> None:
        schedule_source = (REPO_ROOT / "training" / "schedule.py").read_text()
        assert "math.cos" in schedule_source
        assert "Linear warmup" in _text()
        assert "cosine decay" in _text()

    def test_no_unimplemented_flash_attention_claim(self) -> None:
        attention_source = "\n".join(
            path.read_text() for path in (REPO_ROOT / "open_mythos" / "attention").glob("*.py")
        )
        assert "flash_attn" not in attention_source
        assert "Flash Attention 2" not in _text()
        assert "open-mythos[flash]" not in _text()


class TestTrainingDocs:
    def test_entry_point_is_the_non_deprecated_module(self) -> None:
        assert "run_3b" in _text()
        assert "python training/3b_fine_web_edu.py" not in _text()

    def test_training_commands_use_module_form(self) -> None:
        blocks = [b for b in _BASH_BLOCK.findall(_text()) if "training" in b]
        assert blocks, "README documents no training command"
        for block in blocks:
            for line in block.splitlines():
                if "training" in line and "docs/" not in line:
                    assert " -m training." in line, line

    def test_parallelism_claim_matches_the_wrapper_used(self) -> None:
        source = (REPO_ROOT / "training" / "run_3b.py").read_text()
        if "FullyShardedDataParallel" in source and "DistributedDataParallel" not in source:
            assert "FSDP" in _text()
            assert "PyTorch DDP via" not in _text()

    def test_precision_claim_matches_dynamic_scaling(self) -> None:
        entry = (REPO_ROOT / "training" / "run_3b.py").read_text()
        trainer = (REPO_ROOT / "training" / "trainer.py").read_text()
        assert "_make_grad_scaler" in entry
        assert "GradScaler" in trainer
        assert "scaler.scale" in trainer
        assert "scaler.unscale_" in trainer
        assert "dynamic loss scaling" in _text()
        assert "No dynamic loss scaler" not in _text()

    def test_warmup_and_token_budget_match_the_constants(self) -> None:
        from training import run_3b

        text = _text()
        assert f"warmup ({run_3b.WARMUP} steps)" in text
        assert f"{run_3b.TARGET_TOKENS // 1_000_000_000}B tokens" in text
        assert f"`{run_3b.SUBSET}`" in text
        assert callable(run_3b.main)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
