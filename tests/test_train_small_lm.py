import torch

from examples import train_small_lm
from examples.train_small_lm import (
    NoOpScaler,
    PackedTextDataset,
    build_config,
    save_checkpoint,
)

DIRTY_QUOTED_ALPHA = "\u00e2\u20ac\u0153alpha\u00e2\u20ac\u009d"


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {
            "alpha": [1, 2],
            "beta": [3],
            '"alpha"': [4],
            DIRTY_QUOTED_ALPHA: [5],
        }[text]


def test_packed_text_dataset_appends_eos_after_each_sample():
    samples = [{"text": "alpha"}, {"text": "beta"}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=4,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=True,
    )

    assert ds.data.tolist() == [1, 2, 99, 3, 99]


def test_packed_text_dataset_can_disable_eos_append():
    samples = [{"text": "alpha"}, {"text": "beta"}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=2,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=False,
    )

    assert ds.data.tolist() == [1, 2, 3]


def test_packed_text_dataset_cleans_text_before_tokenizing():
    samples = [{"text": DIRTY_QUOTED_ALPHA}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=1,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=True,
        clean_text=True,
    )

    assert ds.data.tolist() == [4, 99]


def test_packed_text_dataset_can_disable_text_cleaning():
    samples = [{"text": DIRTY_QUOTED_ALPHA}]

    ds = PackedTextDataset(
        samples,
        FakeTokenizer(),
        seq_len=1,
        max_tokens=10,
        text_field="text",
        eos_token_id=FakeTokenizer.eos_token_id,
        append_eos=True,
        clean_text=False,
    )

    assert ds.data.tolist() == [5, 99]


def test_build_loss_weights_emphasizes_only_eos_token():
    weights = train_small_lm.build_loss_weights(
        vocab_size=5,
        eos_token_id=2,
        eos_loss_weight=4.0,
        device=torch.device("cpu"),
    )

    assert weights is not None
    assert weights.tolist() == [1.0, 1.0, 4.0, 1.0, 1.0]


def test_build_loss_weights_uses_default_cross_entropy_when_unweighted():
    weights = train_small_lm.build_loss_weights(
        vocab_size=5,
        eos_token_id=2,
        eos_loss_weight=1.0,
        device=torch.device("cpu"),
    )

    assert weights is None


def test_save_checkpoint_records_eos_loss_weight(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "small.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        NoOpScaler(),
        build_config(vocab_size=100, max_seq_len=8),
        tokenizer="gpt2",
        step=12,
        loss=3.4,
        append_eos=True,
        eos_loss_weight=4.0,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert ckpt["append_eos"] is True
    assert ckpt["eos_loss_weight"] == 4.0


def test_validate_init_from_config_allows_only_max_seq_len_change():
    source = train_small_lm.config_to_dict(build_config(vocab_size=100, max_seq_len=128))
    target = build_config(vocab_size=100, max_seq_len=256)

    train_small_lm.validate_init_from_config(source, target)

    source["dim"] = 512
    try:
        train_small_lm.validate_init_from_config(source, target)
    except ValueError as exc:
        assert "dim" in str(exc)
    else:
        raise AssertionError("expected dim mismatch to fail")


def test_load_matching_model_weights_skips_shape_mismatches():
    model = torch.nn.Module()
    model.same = torch.nn.Linear(2, 2)
    model.changed = torch.nn.Linear(3, 3, bias=False)
    source_state = {
        "same.weight": torch.full_like(model.same.weight, 2.0),
        "same.bias": torch.full_like(model.same.bias, 3.0),
        "changed.weight": torch.ones(4, 4),
        "unknown.weight": torch.ones(1),
    }

    result = train_small_lm.load_matching_model_weights(model, source_state)

    assert result.loaded == 2
    assert result.skipped == ["changed.weight", "unknown.weight"]
    assert torch.equal(model.same.weight, torch.full_like(model.same.weight, 2.0))
    assert torch.equal(model.same.bias, torch.full_like(model.same.bias, 3.0))
