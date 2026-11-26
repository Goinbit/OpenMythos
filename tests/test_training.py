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
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

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

    def test_production_checkpoint_roundtrip_and_resume(self, tmp_path):
        from training.checkpoint import load_checkpoint, save_checkpoint

        torch.manual_seed(42)
        cfg = tiny_cfg()
        src = OpenMythos(cfg)
        src_optimizer = torch.optim.AdamW(src.parameters(), lr=1e-3)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))

        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            return loss.detach()

        step(src, src_optimizer)
        save_checkpoint(
            src,
            src_optimizer,
            step=1,
            cfg=cfg,
            vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path),
            ddp=False,
            master=True,
        )
        checkpoint = tmp_path / "step_0000001.pt"
        assert checkpoint.exists()

        torch.manual_seed(7)
        dst = OpenMythos(cfg)
        dst_optimizer = torch.optim.AdamW(dst.parameters(), lr=1e-3)
        restored_step = load_checkpoint(
            dst, dst_optimizer, str(checkpoint), ddp=False, cfg=cfg
        )
        assert restored_step == 1

        for name, tensor in src.state_dict().items():
            assert torch.equal(tensor, dst.state_dict()[name]), name

        def assert_nested_equal(left, right, path="optimizer"):
            if isinstance(left, torch.Tensor):
                assert isinstance(right, torch.Tensor), path
                assert torch.equal(left, right), path
            elif isinstance(left, dict):
                assert left.keys() == right.keys(), path
                for key in left:
                    assert_nested_equal(left[key], right[key], f"{path}.{key}")
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right), path
                for index, (a, b) in enumerate(zip(left, right)):
                    assert_nested_equal(a, b, f"{path}[{index}]")
            else:
                assert left == right, path

        assert_nested_equal(
            src_optimizer.state_dict(), dst_optimizer.state_dict()
        )
        src_loss = step(src, src_optimizer)
        dst_loss = step(dst, dst_optimizer)
        assert torch.equal(src_loss, dst_loss)
        for (name_a, a), (name_b, b) in zip(
            src.named_parameters(), dst.named_parameters()
        ):
            assert name_a == name_b
            assert torch.equal(a, b), name_a

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
# Trainer non-finite guards (P0.2)
# ---------------------------------------------------------------------------


class _PoisonModel(torch.nn.Module):
    def __init__(self, cfg: MythosConfig, poison: str):
        super().__init__()
        self.cfg = cfg
        self.poison = poison
        self.weight = torch.nn.Parameter(torch.ones(cfg.vocab_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = self.weight.view(1, 1, -1).expand(batch, length, -1)
        if self.poison == "loss":
            logits = logits.clone()
            logits.reshape(-1)[0] = float("nan")
        return logits


class TestTrainerNonfiniteGuard:
    def _trainer(self, tmp_path, poison: str, scaler=None):
        from training.trainer import Trainer

        cfg = tiny_cfg()
        model = _PoisonModel(cfg, poison)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        return Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            dataloader=[(input_ids, targets)],
            device="cpu",
            local_rank=0,
            rank=0,
            world_size=1,
            ddp=False,
            master=True,
            amp_ctx=nullcontext(),
            vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path),
            total_steps=1,
            log_every=1,
            ckpt_every=1,
            grad_accum_steps=1,
            warmup_steps=0,
            max_lr=0.1,
            scaler=scaler,
        )

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_nonfinite_loss_blocks_before_backward(self, tmp_path, value):
        trainer = self._trainer(tmp_path, "loss")
        trainer.model.poison = "loss"
        original_forward = trainer.model.forward

        def poisoned_forward(input_ids):
            logits = original_forward(input_ids)
            logits = logits.clone()
            logits.reshape(-1)[0] = value
            return logits

        trainer.model.forward = poisoned_forward
        backward_calls = 0
        original_backward = torch.Tensor.backward

        def counting_backward(tensor, *args, **kwargs):
            nonlocal backward_calls
            backward_calls += 1
            return original_backward(tensor, *args, **kwargs)

        torch.Tensor.backward = counting_backward
        try:
            with pytest.raises(RuntimeError, match="Non-finite loss"):
                trainer.run()
        finally:
            torch.Tensor.backward = original_backward
        assert backward_calls == 0
        assert list(tmp_path.glob("step_*.pt")) == []

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_nonfinite_gradient_blocks_before_optimizer_step(self, tmp_path, value):
        trainer = self._trainer(tmp_path, "finite")
        step_calls = 0
        original_step = trainer.optimizer.step

        def counting_step(*args, **kwargs):
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        trainer.optimizer.step = counting_step
        original_backward = torch.Tensor.backward

        def poisoned_backward(tensor, *args, **kwargs):
            result = original_backward(tensor, *args, **kwargs)
            for parameter in trainer.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.fill_(value)
            return result

        torch.Tensor.backward = poisoned_backward
        try:
            with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
                trainer.run()
        finally:
            torch.Tensor.backward = original_backward
        assert step_calls == 0
        assert list(tmp_path.glob("step_*.pt")) == []

    def test_finite_step_still_runs(self, tmp_path):
        trainer = self._trainer(tmp_path, "finite")
        trainer.run()
        assert len(list(tmp_path.glob("step_*.pt"))) == 1


class TestTrainerGradScaler:
    @staticmethod
    def _wrap_calls(scaler):
        from unittest.mock import Mock

        for name in ("scale", "unscale_", "step", "update"):
            setattr(scaler, name, Mock(wraps=getattr(scaler, name)))
        return scaler

    def test_enabled_scaler_drives_backward_clipping_and_optimizer(self, tmp_path):
        scaler = self._wrap_calls(
            torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
        )
        trainer = TestTrainerNonfiniteGuard()._trainer(
            tmp_path, "finite", scaler=scaler
        )
        trainer.run()

        assert scaler.scale.call_count == 1
        assert scaler.unscale_.call_count == 1
        assert scaler.step.call_count == 1
        assert scaler.update.call_count == 1
        assert len(list(tmp_path.glob("step_*.pt"))) == 1

    def test_disabled_scaler_preserves_plain_optimizer_path(self, tmp_path):
        scaler = self._wrap_calls(torch.amp.GradScaler("cpu", enabled=False))
        trainer = TestTrainerNonfiniteGuard()._trainer(
            tmp_path, "finite", scaler=scaler
        )
        trainer.run()

        assert scaler.scale.call_count == 0
        assert scaler.unscale_.call_count == 0
        assert scaler.step.call_count == 0
        assert scaler.update.call_count == 0
        assert len(list(tmp_path.glob("step_*.pt"))) == 1

    def test_nonfinite_gradient_blocks_scaled_optimizer_step(self, tmp_path):
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        trainer = TestTrainerNonfiniteGuard()._trainer(
            tmp_path, "finite", scaler=scaler
        )
        step_calls = 0
        original_step = trainer.optimizer.step

        def counting_step(*args, **kwargs):
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        trainer.optimizer.step = counting_step
        original_backward = torch.Tensor.backward

        def poisoned_backward(tensor, *args, **kwargs):
            result = original_backward(tensor, *args, **kwargs)
            for parameter in trainer.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.fill_(float("inf"))
            return result

        torch.Tensor.backward = poisoned_backward
        try:
            with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
                trainer.run()
        finally:
            torch.Tensor.backward = original_backward
        assert step_calls == 0
        assert list(tmp_path.glob("step_*.pt")) == []

    def test_checkpoint_roundtrip_preserves_scaler_state(self, tmp_path):
        from training.checkpoint import load_checkpoint, save_checkpoint

        cfg = tiny_cfg()
        source_model = _PoisonModel(cfg, "finite")
        source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1)
        source_scaler = torch.amp.GradScaler("cpu", enabled=True)
        source_scaler.load_state_dict(
            {
                "scale": 4096.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.25,
                "growth_interval": 500,
                "_growth_tracker": 17,
            }
        )
        save_checkpoint(
            source_model,
            source_optimizer,
            step=7,
            cfg=cfg,
            vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path),
            ddp=False,
            master=True,
            scaler=source_scaler,
        )

        target_model = _PoisonModel(cfg, "finite")
        target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1)
        target_scaler = torch.amp.GradScaler("cpu", enabled=True)
        step = load_checkpoint(
            target_model,
            target_optimizer,
            str(tmp_path / "step_0000007.pt"),
            ddp=False,
            cfg=cfg,
            scaler=target_scaler,
        )

        assert step == 7
        assert target_scaler.state_dict() == source_scaler.state_dict()

    def test_entry_point_enables_scaler_only_for_fp16(self):
        source = (
            Path(__file__).resolve().parent.parent / "training" / "run_3b.py"
        ).read_text()
        assert 'torch.amp.GradScaler("cuda", enabled=' in source
        assert "dt == torch.float16" in source
        assert "scaler=scaler" in source


# ---------------------------------------------------------------------------
# Held-out validation + distributed eval (P4.5)
# ---------------------------------------------------------------------------


class _FakeStreamingDataset:
    def __init__(self, samples):
        self.samples = samples

    def shard(self, num_shards: int, index: int):
        return [sample for i, sample in enumerate(self.samples) if i % num_shards == index]


class TestHeldoutEval:
    def test_document_hash_partition_is_stable_and_bounded(self):
        from training.data import _is_validation_document

        sample = {"id": "stable-document-id", "text": "ignored"}
        assert _is_validation_document(sample, 0.01) == _is_validation_document(sample, 0.01)
        assert not _is_validation_document(sample, 0.0)
        assert _is_validation_document(sample, 1.0)

    def test_train_and_validation_documents_are_disjoint(self, monkeypatch):
        import training.data as data_module

        samples = [
            {"id": "train-a", "text": "aaaa"},
            {"id": "val-a", "text": "bbbb"},
            {"id": "train-b", "text": "cccc"},
            {"id": "val-b", "text": "dddd"},
        ]
        monkeypatch.setattr(data_module, "load_dataset", lambda *args, **kwargs: _FakeStreamingDataset(samples))
        monkeypatch.setattr(
            data_module,
            "_is_validation_document",
            lambda sample, fraction: sample["id"].startswith("val-"),
        )
        class _Encoding:
            @staticmethod
            def encode(text):
                return [ord(char) for char in text]

        train = data_module.FineWebEduDataset(_Encoding(), 3, "sample-10BT", 0, 1, split="train")
        validation = data_module.FineWebEduDataset(_Encoding(), 3, "sample-10BT", 0, 1, split="validation")

        train_tokens = [pair[0].tolist() for pair in train]
        validation_tokens = [pair[0].tolist() for pair in validation]
        assert train_tokens == [[97, 97, 97], [99, 99, 99]]
        assert validation_tokens == [[98, 98, 98], [100, 100, 100]]
        assert {tuple(tokens) for tokens in train_tokens}.isdisjoint(
            tuple(tokens) for tokens in validation_tokens
        )

    def test_validation_loader_marks_dataset_as_heldout(self):
        from training.eval import build_validation_loader

        loader = build_validation_loader(lambda text: [], 0, 1, 16, num_workers=0)
        assert loader.dataset.split == "validation"

    def test_distributed_eval_reduces_loss_and_count(self, monkeypatch):
        import training.eval as eval_module

        class _EvalModel(torch.nn.Module):
            def forward(self, input_ids):
                batch, length = input_ids.shape
                return torch.zeros(batch, length, 4)

        model = _EvalModel().train()
        loader = [(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long))]
        reduce_calls = 0

        def fake_all_reduce(tensor, op):
            nonlocal reduce_calls
            reduce_calls += 1
            assert op == eval_module.dist.ReduceOp.SUM
            tensor[0] = 6.0
            tensor[1] = 2.0

        monkeypatch.setattr(eval_module.dist, "all_reduce", fake_all_reduce)
        loss = eval_module.compute_val_loss(model, loader, "cpu", 4, num_batches=1, ddp=True)
        assert loss == pytest.approx(3.0)
        assert reduce_calls == 1
        assert model.training

    def test_nonmaster_rank_still_participates_in_scheduled_eval(self, monkeypatch):
        import training.eval as eval_module

        calls = 0

        def fake_compute(*args, **kwargs):
            nonlocal calls
            calls += 1
            return 1.25

        monkeypatch.setattr(eval_module, "compute_val_loss", fake_compute)
        trainer = type("TrainerStub", (), {"model": object(), "master": False, "ddp": True})()
        hook = eval_module.PeriodicEvalHook(10, [], "cpu", 4)
        hook(10, trainer)
        assert calls == 1


class _BalanceModel(torch.nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.ones(cfg.vocab_size))
        self.balance_scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, input_ids: torch.Tensor, return_balance_loss: bool = False):
        batch, length = input_ids.shape
        logits = self.weight.view(1, 1, -1).expand(batch, length, -1)
        if return_balance_loss:
            return logits, self.cfg.moe_balance_alpha * self.balance_scale.square()
        return logits


class TestTrainerBalanceLoss:
    def test_trainer_adds_balance_loss_when_configured(self, tmp_path):
        from training.trainer import Trainer

        cfg = tiny_cfg(moe_balance_alpha=0.1)
        model = _BalanceModel(cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        trainer = Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            dataloader=[(input_ids, targets)],
            device="cpu",
            local_rank=0,
            rank=0,
            world_size=1,
            ddp=False,
            master=True,
            amp_ctx=nullcontext(),
            vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path),
            total_steps=1,
            log_every=1,
            ckpt_every=1,
            grad_accum_steps=1,
            warmup_steps=0,
            max_lr=0.1,
            grad_clip=100.0,
        )
        trainer.run()
        assert model.balance_scale.grad is not None
        assert model.balance_scale.grad.item() == pytest.approx(0.2)

    def test_trainer_skips_balance_loss_when_alpha_is_zero(self, tmp_path):
        from training.trainer import Trainer

        cfg = tiny_cfg(moe_balance_alpha=0.0)
        model = _BalanceModel(cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        trainer = Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            dataloader=[(input_ids, targets)],
            device="cpu",
            local_rank=0,
            rank=0,
            world_size=1,
            ddp=False,
            master=True,
            amp_ctx=nullcontext(),
            vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path),
            total_steps=1,
            log_every=1,
            ckpt_every=1,
            grad_accum_steps=1,
            warmup_steps=0,
            max_lr=0.1,
        )
        trainer.run()
        assert model.balance_scale.grad is None


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


class _SeededIterableDataset(IterableDataset):
    def __iter__(self):
        for _ in range(2):
            yield torch.randint(0, 1000, (4,))


class TestDataLoaderReproducibility:
    def test_worker_init_fn_uses_worker_specific_seed(self, monkeypatch):
        import training.data as data_module

        random_seeds = []
        torch_seeds = []
        monkeypatch.setattr(data_module.random, "seed", random_seeds.append)
        monkeypatch.setattr(data_module.torch, "manual_seed", torch_seeds.append)
        monkeypatch.setattr(data_module.torch, "initial_seed", lambda: 42)

        data_module._seed_worker(3)

        assert random_seeds == [45]
        assert torch_seeds == [45]

    def test_production_loader_uses_worker_seed_fn(self):
        from training.data import _seed_worker, build_dataloader

        class _Encoding:
            @staticmethod
            def encode(text):
                return []

        loader = build_dataloader(
            _Encoding(), 16, "sample-10BT", 0, 1, 2, num_workers=0
        )
        assert loader.worker_init_fn is _seed_worker

    def test_real_multiworker_loader_reproduces_batches(self):
        from training.data import _seed_worker

        def collect(seed: int):
            torch.manual_seed(seed)
            loader = DataLoader(
                _SeededIterableDataset(),
                batch_size=2,
                num_workers=2,
                worker_init_fn=_seed_worker,
            )
            return list(loader)

        first = collect(123)
        second = collect(123)
        different = collect(124)
        assert len(first) == len(second) == 2
        for a, b in zip(first, second):
            assert torch.equal(a, b)
        assert any(not torch.equal(a, b) for a, b in zip(first, different))


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
        # P4.5: training script was split across training/*.py in Stage 4.5.
        # Check the new entry point (run_3b.py) where cfg construction lives.
        train_dir = Path(__file__).resolve().parent.parent / "training"
        sources = "\n".join(
            (train_dir / name).read_text()
            for name in ("run_3b.py", "3b_fine_web_edu.py")
        )
        assert "cfg.vocab_size =" not in sources, (
            "training script must use dataclasses.replace, not mutation"
        )
        assert "cfg.max_seq_len =" not in sources, (
            "training script must use dataclasses.replace, not mutation"
        )
        assert (
            "from dataclasses import replace" in sources
            or "dataclasses.replace" in sources
        )


# ---------------------------------------------------------------------------
# P3.8: training-loop micro-optimizations (source-level guards)
# ---------------------------------------------------------------------------


class TestTrainingOptimizations:
    # P4.5: training script split across training/*.py in Stage 4.5. Each
    # property below now lives in the specific module that owns that concern.

    def _training_src(self) -> str:
        train_dir = Path(__file__).resolve().parent.parent / "training"
        return "\n".join(
            (train_dir / name).read_text()
            for name in (
                "3b_fine_web_edu.py",
                "run_3b.py",
                "trainer.py",
                "data.py",
                "constants.py",
                "schedule.py",
                "eval.py",
            )
        )

    def test_zero_grad_uses_set_to_none(self):
        src = self._training_src()
        assert "zero_grad()" not in src, "zero_grad() without set_to_none=True found"
        assert "zero_grad(set_to_none=True)" in src

    def test_dataset_buffer_no_copy_slice(self):
        src = self._training_src()
        # The old O(n) slice pattern should not be used
        assert "buf = buf[" not in src, "O(n) list slice pattern still present"
        # Either deque or head-index must be present
        assert "deque" in src or "head +=" in src or "head_idx" in src

    def test_gradient_checkpointing_flag_exists(self):
        src = self._training_src()
        assert "USE_GRAD_CHECKPOINT" in src or "gradient_checkpointing" in src.lower()


# ---------------------------------------------------------------------------
# P4.5: Training determinism regression gate
# ---------------------------------------------------------------------------


class TestTrainingDeterminism:
    """Stage 4.5: same seed + same batches must produce same loss curve.

    Regression gate catching silent seed leaks in DataLoader workers,
    Trainer state, or checkpoint resume paths.
    """

    def _run_training(self, seed: int, n_steps: int = 10) -> list:
        """Runs n_steps on a fixed tiny model with deterministic batches.

        Returns the list of per-step loss values.
        """
        torch.manual_seed(seed)
        cfg = tiny_cfg()

        torch.manual_seed(seed)
        model = OpenMythos(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(seed + 1000)  # distinct seed for data
        fixed_batches = [
            (
                torch.randint(0, cfg.vocab_size, (2, 8)),
                torch.randint(0, cfg.vocab_size, (2, 8)),
            )
            for _ in range(n_steps)
        ]

        losses = []
        model.train()
        for x, y in fixed_batches:
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    def test_same_seed_same_loss_curve(self):
        """Two fresh runs with the same seed must produce identical losses."""
        run1 = self._run_training(seed=42, n_steps=10)
        run2 = self._run_training(seed=42, n_steps=10)

        t1 = torch.tensor(run1)
        t2 = torch.tensor(run2)
        torch.testing.assert_close(t1, t2, atol=1e-5, rtol=1e-5)

    def test_different_seed_different_loss_curve(self):
        """Sanity: the determinism test is meaningful (different seeds diverge)."""
        run_a = self._run_training(seed=42, n_steps=5)
        run_b = self._run_training(seed=77, n_steps=5)
        assert run_a != run_b, (
            "different seeds produced identical curves — test is broken"
        )


# ---------------------------------------------------------------------------
# P3.8 completion: activation checkpointing wrap on RecurrentBlock
# ---------------------------------------------------------------------------


class TestGradCheckpointWrap:
    """Verify ``apply_grad_checkpoint_if_enabled`` is a correct no-op when
    USE_GRAD_CHECKPOINT is False, and wraps every RecurrentBlock's forward
    when True, without breaking gradient flow or changing numerics beyond
    the expected recompute non-determinism."""

    def test_noop_when_disabled(self):
        """Default USE_GRAD_CHECKPOINT=False must leave model.forward untouched.

        Bound-method identity is unstable in Python, so compare instance
        ``__dict__`` to confirm no wrapper was installed.
        """
        from training.trainer import apply_grad_checkpoint_if_enabled

        torch.manual_seed(42)
        model = OpenMythos(tiny_cfg())
        n = apply_grad_checkpoint_if_enabled(model)
        assert n == 0
        # Wrapping installs `forward` onto the instance __dict__ shadowing
        # the class method; disabled path must NOT do that.
        assert "forward" not in model.recurrent.__dict__

    def test_wrapping_preserves_forward_equivalence(self, monkeypatch):
        """With grad_ckpt on, forward output must equal forward without
        (same seed; recompute does not alter deterministic pure-compute path)."""
        import training.trainer as trainer_mod
        from training.trainer import apply_grad_checkpoint_if_enabled

        torch.manual_seed(42)
        cfg = tiny_cfg()
        model_a = OpenMythos(cfg)
        model_b = OpenMythos(cfg)
        # sync params
        model_b.load_state_dict(model_a.state_dict())

        monkeypatch.setattr(trainer_mod, "USE_GRAD_CHECKPOINT", True)
        n = apply_grad_checkpoint_if_enabled(model_b)
        assert n >= 1  # at least one RecurrentBlock wrapped

        x = torch.randint(0, cfg.vocab_size, (2, 8))
        out_a = model_a(x)
        out_b = model_b(x)
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)

    def test_wrapping_preserves_gradient_flow(self, monkeypatch):
        """Backward through a checkpointed RecurrentBlock must still populate
        every trainable parameter's .grad (use_reentrant=False is required)."""
        import training.trainer as trainer_mod
        from training.trainer import apply_grad_checkpoint_if_enabled

        torch.manual_seed(42)
        cfg = tiny_cfg()
        model = OpenMythos(cfg)
        monkeypatch.setattr(trainer_mod, "USE_GRAD_CHECKPOINT", True)
        apply_grad_checkpoint_if_enabled(model)

        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits = model(x)
        # Target is a second pseudo-batch so loss != 0.
        target = torch.randint(0, cfg.vocab_size, (2, 8))
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.reshape(-1))
        loss.backward()

        recurrent_params = dict(model.recurrent.named_parameters())
        missing = [
            name for name, p in recurrent_params.items()
            if p.requires_grad and p.grad is None
        ]
        # MoE experts may miss grads for tokens not routed to them (same as
        # baseline moda.py gradient_flow test). Tolerate but bound.
        assert len(missing) <= len(recurrent_params) // 2, (
            f"too many recurrent params missing grads under checkpointing: "
            f"{len(missing)}/{len(recurrent_params)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
