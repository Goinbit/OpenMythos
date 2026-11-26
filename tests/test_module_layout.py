"""P4.1 module layout and backward-compatible import contract."""
from __future__ import annotations

import importlib
from pathlib import Path

import open_mythos
from open_mythos import MythosConfig, OpenMythos
from open_mythos.main import MoEFFN as LegacyMoEFFN


def _cfg() -> MythosConfig:
    return MythosConfig(
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
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )


def test_moe_package_exports_preserve_public_imports() -> None:
    package = importlib.import_module("open_mythos.moe")
    ffn = importlib.import_module("open_mythos.moe.ffn")
    router = importlib.import_module("open_mythos.moe.router")
    assert package.MoEFFN is ffn.MoEFFN is LegacyMoEFFN
    assert package.BiasRouter is router.BiasRouter
    assert package.RouterProtocol is router.RouterProtocol
    assert package.build_router is router.build_router


def test_generate_module_exports_the_model_generator() -> None:
    module = importlib.import_module("open_mythos.generate")
    assert module.generate is OpenMythos.generate


def test_module_split_preserves_state_dict_layout() -> None:
    model = OpenMythos(_cfg())
    keys = set(model.state_dict())
    assert "recurrent.block.ffn.router.linear.weight" in keys
    assert "recurrent.block.ffn.router.router_bias" in keys
    assert "recurrent.block.ffn.routed_experts.gate_weight" in keys
    assert "recurrent.block.ffn.routed_experts.up_weight" in keys
    assert "recurrent.block.ffn.routed_experts.down_weight" in keys


def test_core_modules_stay_under_400_lines() -> None:
    root = Path(open_mythos.__file__).parent
    oversized = {
        str(path.relative_to(root)): len(path.read_text().splitlines())
        for path in root.rglob("*.py")
        if "experimental" not in path.parts
        and len(path.read_text().splitlines()) > 400
    }
    assert oversized == {}
