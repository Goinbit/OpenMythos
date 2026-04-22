"""
Training concerns testable without GPU, FSDP, or dataset streaming.

Covers:
    - single-batch overfit (loss decreases)
    - checkpoint state_dict roundtrip
    - checkpoint non-finite param guard (P0.2)
    - LR schedule shape (warmup + cosine)
    - seed-determined init (P0.5)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from open_mythos.main import MythosConfig, OpenMythos

# ---------------------------------------------------------------------------
# Load the training script as a module (filename starts with a digit so it
# isn't a normal import target). If any heavy top-level import is unavailable
# in this environment, skip cleanly.
# ---------------------------------------------------------------------------

_TRAIN_SCRIPT = (
    Path(__file__).resolve().parent.parent / "training" / "3b_fine_web_edu.py"
)

try:  # pragma: no cover - exercised via skip path
    _spec = importlib.util.spec_from_file_location("train_3b", _TRAIN_SCRIPT)
    _train_mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_train_mod)
    _TRAIN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any failure means skip
    _train_mod = None
    _TRAIN_IMPORT_ERROR = exc


def _require_train_mod():
    if _train_mod is None:
        pytest.skip(f"training script not importable: {_TRAIN_IMPORT_ERROR}")
    return _train_mod


# ---------------------------------------------------------------------------
# Tiny config shared across tests
# ---------------------------------------------------------------------------


def tiny_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=100,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        # MLA fields required even when unused
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


# ---------------------------------------------------------------------------
# Overfit a single batch — learning signal works end-to-end
# ---------------------------------------------------------------------------


class TestOverfitSingleBatch:
    def test_loss_decreases_on_single_batch(self):
        torch.manual_seed(42)
        cfg = tiny_cfg()
        model = OpenMythos(cfg)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        inp, tgt = ids[:, :-1], ids[:, 1:]

        def loss_fn() -> torch.Tensor:
            logits = model(inp)
            return F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)
            )

        model.train()
        initial = loss_fn().item()
        for _ in range(10):
            optim.zero_grad()
            loss = loss_fn()
            loss.backward()
            optim.step()
        final = loss_fn().item()

        assert final < initial * 0.95, (
            f"loss did not drop enough: initial={initial:.4f} final={final:.4f}"
        )


# ---------------------------------------------------------------------------
# Checkpoint round-trip and non-finite guard
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_state_dict_roundtrip(self, tmp_path):
        torch.manual_seed(42)
        cfg = tiny_cfg()
        src = OpenMythos(cfg)

        ckpt_path = tmp_path / "model.pt"
        torch.save(src.state_dict(), ckpt_path)

        torch.manual_seed(7)  # deliberately different seed so init differs
        dst = OpenMythos(cfg)
        dst.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        src_state, dst_state = src.state_dict(), dst.state_dict()
        assert src_state.keys() == dst_state.keys()
        for name in src_state:
            assert torch.equal(src_state[name], dst_state[name]), (
                f"tensor mismatch after roundtrip: {name}"
            )

        # Buffers are part of state_dict; sanity-check the ones P0/P1 care about.
        assert "freqs_cis" in src_state
        for name, _ in src.named_buffers():
            assert torch.equal(src_state[name], dst_state[name]), (
                f"buffer mismatch after roundtrip: {name}"
            )

    def test_checkpoint_refuses_nonfinite_params(self, tmp_path):
        mod = _require_train_mod()
        if not hasattr(mod, "save_checkpoint"):
            pytest.skip("save_checkpoint not defined in training script")

        torch.manual_seed(42)
        cfg = tiny_cfg()
        model = OpenMythos(cfg)
        with torch.no_grad():
            next(iter(model.parameters())).fill_(float("nan"))

        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with pytest.raises(RuntimeError, match="Non-finite parameter"):
            mod.save_checkpoint(
                model,
                optim,
                step=0,
                cfg=cfg,
                vocab_size=cfg.vocab_size,
                ckpt_dir=str(tmp_path),
                ddp=False,
                master=True,
            )


# ---------------------------------------------------------------------------
# LR schedule: warmup then cosine decay to min_lr
# ---------------------------------------------------------------------------


class TestTrainingUtilities:
    def test_get_lr_warmup_then_cosine(self):
        mod = _require_train_mod()
        if not hasattr(mod, "get_lr"):
            pytest.skip("get_lr not defined in training script")

        warmup, total = 100, 1000
        max_lr, min_lr = 1e-3, 1e-4

        lr_start = mod.get_lr(0, warmup, total, max_lr, min_lr)
        lr_peak = mod.get_lr(warmup, warmup, total, max_lr, min_lr)
        lr_end = mod.get_lr(total, warmup, total, max_lr, min_lr)
        lr_mid_warmup = mod.get_lr(warmup // 2, warmup, total, max_lr, min_lr)

        assert lr_start == pytest.approx(0.0, abs=1e-9)
        assert lr_peak == pytest.approx(max_lr, rel=1e-6)
        assert lr_end == pytest.approx(min_lr, rel=1e-6)
        assert 0 < lr_mid_warmup < max_lr
        # min_lr is 10% of max_lr by convention in this training script
        assert lr_end == pytest.approx(max_lr * 0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# Seed reproducibility — documents the property the training script relies on
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    def test_same_seed_same_init(self):
        cfg = tiny_cfg()

        torch.manual_seed(42)
        a = OpenMythos(cfg)
        snap_a = next(iter(a.parameters())).detach().clone()

        torch.manual_seed(42)
        b = OpenMythos(cfg)
        snap_b = next(iter(b.parameters())).detach().clone()

        assert torch.equal(snap_a, snap_b)


# ---------------------------------------------------------------------------
# P1.4: frozen-config + vocab_size consistency guards
# ---------------------------------------------------------------------------


class TestVocabSizeValidation:
    def test_tokenizer_model_vocab_mismatch_raises(self):
        mod = _require_train_mod()
        if not hasattr(mod, "_validate_vocab_compatibility"):
            pytest.skip("_validate_vocab_compatibility not defined")

        class _FakeEncoding:
            vocab_size = 12345

        cfg = tiny_cfg(vocab_size=100)
        with pytest.raises(ValueError, match=r"12345.*100|100.*12345"):
            mod._validate_vocab_compatibility(_FakeEncoding(), cfg)

    def test_checkpoint_vocab_mismatch_rejected(self, tmp_path):
        mod = _require_train_mod()
        if not hasattr(mod, "load_checkpoint"):
            pytest.skip("load_checkpoint not defined")

        torch.manual_seed(42)
        cfg_saved = tiny_cfg(vocab_size=100)
        model = OpenMythos(cfg_saved)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ckpt_path = tmp_path / "step_0000000.pt"
        torch.save(
            {
                "step": 0,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "cfg": cfg_saved,
                "vocab_size": 100,
            },
            ckpt_path,
        )

        cfg_loaded = tiny_cfg(vocab_size=50)
        model_loaded = OpenMythos(cfg_loaded)
        optim_loaded = torch.optim.AdamW(model_loaded.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match=r"100.*50|50.*100"):
            mod.load_checkpoint(
                model_loaded, optim_loaded, str(ckpt_path), ddp=False, cfg=cfg_loaded
            )

    def test_dataclasses_replace_is_used(self):
        train_script_path = (
            Path(__file__).resolve().parent.parent / "training" / "3b_fine_web_edu.py"
        )
        source = train_script_path.read_text()
        assert "cfg.vocab_size =" not in source, (
            "training script must use dataclasses.replace, not mutation"
        )
        assert "cfg.max_seq_len =" not in source, (
            "training script must use dataclasses.replace, not mutation"
        )
        assert "from dataclasses import replace" in source or "dataclasses.replace" in source


# ---------------------------------------------------------------------------
# P3.8: training-loop micro-optimizations (source-level guards)
# ---------------------------------------------------------------------------


class TestTrainingOptimizations:
    def test_zero_grad_uses_set_to_none(self):
        src = (
            Path(__file__).resolve().parent.parent / "training" / "3b_fine_web_edu.py"
        ).read_text()
        assert "zero_grad()" not in src, "zero_grad() without set_to_none=True found"
        assert "zero_grad(set_to_none=True)" in src

    def test_dataset_buffer_no_copy_slice(self):
        src = (
            Path(__file__).resolve().parent.parent / "training" / "3b_fine_web_edu.py"
        ).read_text()
        # The old O(n) slice pattern should not be used
        assert "buf = buf[" not in src, "O(n) list slice pattern still present"
        # Either deque or head-index must be present
        assert "deque" in src or "head +=" in src or "head_idx" in src

    def test_gradient_checkpointing_flag_exists(self):
        src = (
            Path(__file__).resolve().parent.parent / "training" / "3b_fine_web_edu.py"
        ).read_text()
        assert "USE_GRAD_CHECKPOINT" in src or "gradient_checkpointing" in src.lower()


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
