"""Train a small OpenMythos language model with a Hugging Face tokenizer.

This is the practical next step after the tiny byte-level smoke example:
it trains OpenMythos itself on a real text dataset, saves a reusable
checkpoint, and pairs with ``examples/generate_small_lm.py`` for inference.

CPU smoke:
    python -m examples.train_small_lm --steps 20 --batch-size 4 --seq-len 64

Longer local run:
    python -m examples.train_small_lm --steps 2000 --device cuda
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from examples.small_lm_text import clean_text_artifacts
from open_mythos import MythosConfig, OpenMythos


@dataclass(frozen=True)
class WeightLoadResult:
    loaded: int
    skipped: list[str]


class NoOpScaler:
    """Keep the train step identical when AMP is disabled."""

    @staticmethod
    def scale(loss: torch.Tensor) -> torch.Tensor:
        return loss

    @staticmethod
    def step(optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    @staticmethod
    def update() -> None:
        return None

    @staticmethod
    def state_dict() -> dict[str, Any]:
        return {}

    @staticmethod
    def load_state_dict(_state: dict[str, Any]) -> None:
        return None


class PackedTextDataset(Dataset):
    """Materialize streamed text into fixed-length next-token pairs."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_len: int,
        max_tokens: int,
        text_field: str,
        eos_token_id: int | None,
        append_eos: bool,
        clean_text: bool = True,
    ) -> None:
        if append_eos and eos_token_id is None:
            raise ValueError(
                "append_eos=True but tokenizer has no eos_token_id; "
                "pass --no-append-eos or choose a tokenizer with EOS"
            )
        buf: list[int] = []
        for sample in hf_dataset:
            text = sample[text_field]
            if isinstance(text, str) and text.strip():
                if clean_text:
                    text = clean_text_artifacts(text)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if append_eos and eos_token_id is not None:
                    ids.append(eos_token_id)
                buf.extend(ids)
            if len(buf) >= max_tokens:
                break
        if len(buf) <= seq_len:
            raise ValueError(
                f"dataset produced only {len(buf)} tokens; need > seq_len={seq_len}"
            )
        n_pairs = max(1, (len(buf) - 1) // seq_len)
        self.seq_len = seq_len
        self.data = torch.tensor(buf[: n_pairs * seq_len + 1], dtype=torch.long)

    def __len__(self) -> int:
        return (self.data.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_text_dataset(name: str, config: str, split: str):
    if config:
        return load_dataset(name, config, split=split, streaming=True)
    return load_dataset(name, split=split, streaming=True)


def build_scaler(use_amp: bool):
    if not use_amp:
        return NoOpScaler()
    amp_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if amp_scaler is not None:
        return amp_scaler("cuda", enabled=True)
    torch_scaler = getattr(torch, "GradScaler", None)
    if torch_scaler is not None:
        return torch_scaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def build_config(vocab_size: int, max_seq_len: int) -> MythosConfig:
    return MythosConfig(
        vocab_size=vocab_size,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=max_seq_len,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=256,
        lora_rank=8,
        dropout=0.0,
    )


def config_to_dict(cfg: MythosConfig) -> dict[str, int | float | str]:
    return {
        "vocab_size": cfg.vocab_size,
        "dim": cfg.dim,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "max_seq_len": cfg.max_seq_len,
        "max_loop_iters": cfg.max_loop_iters,
        "prelude_layers": cfg.prelude_layers,
        "coda_layers": cfg.coda_layers,
        "attn_type": cfg.attn_type,
        "n_experts": cfg.n_experts,
        "n_shared_experts": cfg.n_shared_experts,
        "n_experts_per_tok": cfg.n_experts_per_tok,
        "expert_dim": cfg.expert_dim,
        "lora_rank": cfg.lora_rank,
        "dropout": cfg.dropout,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    cfg: MythosConfig,
    tokenizer: str,
    step: int,
    loss: float,
    append_eos: bool,
    eos_loss_weight: float,
    clean_text: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config_to_dict(cfg),
            "tokenizer": tokenizer,
            "step": step,
            "loss": loss,
            "append_eos": append_eos,
            "eos_loss_weight": eos_loss_weight,
            "clean_text": clean_text,
        },
        path,
    )
    print(f"[saved] {path} (step={step}, loss={loss:.4f})")


def build_loss_weights(
    vocab_size: int,
    eos_token_id: int | None,
    eos_loss_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    if eos_loss_weight <= 0:
        raise ValueError(f"--eos-loss-weight must be positive, got {eos_loss_weight}")
    if eos_loss_weight == 1.0:
        return None
    if eos_token_id is None:
        raise ValueError("--eos-loss-weight requires a tokenizer with eos_token_id")
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ValueError(
            f"eos_token_id={eos_token_id} is outside vocab_size={vocab_size}"
        )
    weights = torch.ones(vocab_size, dtype=torch.float32, device=device)
    weights[eos_token_id] = eos_loss_weight
    return weights


def validate_init_from_config(
    source_config: dict[str, Any],
    target_cfg: MythosConfig,
) -> None:
    target_config = config_to_dict(target_cfg)
    mismatches = [
        key
        for key, value in source_config.items()
        if key != "max_seq_len" and target_config.get(key) != value
    ]
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ValueError(f"init-from checkpoint config mismatch: {fields}")


def load_matching_model_weights(
    model: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
) -> WeightLoadResult:
    target_state = model.state_dict()
    loadable = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    skipped = sorted(key for key in source_state if key not in loadable)
    model.load_state_dict(loadable, strict=False)
    return WeightLoadResult(loaded=len(loadable), skipped=skipped)


@torch.no_grad()
def evaluate(
    model: OpenMythos,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    n_loops: int,
    batches: int,
) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--train-tokens", type=int, default=500_000)
    parser.add_argument("--eval-tokens", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init-from",
        default="",
        help=(
            "warm-start model weights from a checkpoint without loading optimizer; "
            "allows max_seq_len changes"
        ),
    )
    parser.add_argument(
        "--no-append-eos",
        action="store_true",
        help="do not append tokenizer.eos_token_id after each text sample",
    )
    parser.add_argument(
        "--eos-loss-weight",
        type=float,
        default=1.0,
        help="multiply cross-entropy loss for EOS targets; >1 encourages natural stops",
    )
    parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="disable deterministic cleanup of mojibake punctuation before tokenizing",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use CUDA autocast + GradScaler when training on cuda",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--out", default="checkpoints/small_mythos.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")

    print(
        f"[setup] device={device} batch={args.batch_size} "
        f"seq_len={args.seq_len} steps={args.steps}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    cfg = build_config(vocab_size, args.seq_len)
    append_eos = not args.no_append_eos
    clean_text = not args.no_clean_text
    if not append_eos and args.eos_loss_weight != 1.0:
        raise ValueError("--eos-loss-weight > 1 requires EOS labels; remove --no-append-eos")
    loss_weights = build_loss_weights(
        vocab_size,
        tokenizer.eos_token_id,
        args.eos_loss_weight,
        device,
    )

    print(f"[setup] tokenizer={args.tokenizer} vocab_size={vocab_size:,}")
    print(
        f"[setup] dataset={args.dataset} config={args.dataset_config or 'none'} "
        f"append_eos={append_eos} eos_loss_weight={args.eos_loss_weight} "
        f"clean_text={clean_text}"
    )
    raw_train = load_text_dataset(args.dataset, args.dataset_config, args.train_split)
    raw_eval = load_text_dataset(args.dataset, args.dataset_config, args.eval_split)
    train_ds = PackedTextDataset(
        raw_train,
        tokenizer,
        args.seq_len,
        args.train_tokens,
        args.text_field,
        tokenizer.eos_token_id,
        append_eos,
        clean_text=clean_text,
    )
    eval_ds = PackedTextDataset(
        raw_eval,
        tokenizer,
        args.seq_len,
        args.eval_tokens,
        args.text_field,
        tokenizer.eos_token_id,
        append_eos,
        clean_text=clean_text,
    )
    print(
        f"[setup] train tokens={train_ds.data.numel():,} pairs={len(train_ds):,} | "
        f"eval tokens={eval_ds.data.numel():,} pairs={len(eval_ds):,}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    data_iter = iter(train_loader)

    model = OpenMythos(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = build_scaler(use_amp)
    start_step = 0
    last_loss = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["config"] != config_to_dict(cfg):
            raise ValueError("resume checkpoint config does not match current run")
        if "append_eos" in ckpt and bool(ckpt["append_eos"]) != append_eos:
            raise ValueError("resume checkpoint append_eos does not match current run")
        if "clean_text" in ckpt and bool(ckpt["clean_text"]) != clean_text:
            raise ValueError("resume checkpoint clean_text does not match current run")
        if (
            "eos_loss_weight" in ckpt
            and float(ckpt["eos_loss_weight"]) != args.eos_loss_weight
        ):
            raise ValueError(
                "resume checkpoint eos_loss_weight does not match current run"
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        last_loss = float(ckpt.get("loss", 0.0))
        print(f"[resume] {args.resume} at step={start_step} loss={last_loss:.4f}")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        validate_init_from_config(ckpt["config"], cfg)
        if "append_eos" in ckpt and bool(ckpt["append_eos"]) != append_eos:
            raise ValueError("init-from checkpoint append_eos does not match current run")
        if "clean_text" in ckpt and bool(ckpt["clean_text"]) != clean_text:
            raise ValueError("init-from checkpoint clean_text does not match current run")
        if (
            "eos_loss_weight" in ckpt
            and float(ckpt["eos_loss_weight"]) != args.eos_loss_weight
        ):
            raise ValueError(
                "init-from checkpoint eos_loss_weight does not match current run"
            )
        result = load_matching_model_weights(model, ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        last_loss = float(ckpt.get("loss", 0.0))
        skipped = ", ".join(result.skipped) if result.skipped else "none"
        print(
            f"[init-from] {args.init_from} at step={start_step} "
            f"loss={last_loss:.4f}; loaded={result.loaded} skipped={skipped}"
        )
    params = sum(p.numel() for p in model.parameters())
    print(f"[setup] OpenMythos params={params:,} amp={use_amp}")

    model.train()
    start_time = time.perf_counter()
    out = Path(args.out)
    final_step = start_step + args.steps
    last_saved_step = -1
    for step in range(start_step + 1, final_step + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(x, n_loops=args.n_loops)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
                weight=loss_weights,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.item())

        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            completed = step - start_step
            tok_s = (completed * x.numel()) / max(elapsed, 1e-9)
            print(f"step {step:5d} | train loss {last_loss:.4f} | tok/s {tok_s:,.0f}")
        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = evaluate(
                model,
                eval_loader,
                device,
                vocab_size,
                args.n_loops,
                args.eval_batches,
            )
            print(f"           | val loss   {val_loss:.4f}")
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                out,
                model,
                optimizer,
                scaler,
                cfg,
                args.tokenizer,
                step,
                last_loss,
                append_eos,
                args.eos_loss_weight,
                clean_text,
            )
            last_saved_step = step

    if last_saved_step != final_step:
        save_checkpoint(
            out,
            model,
            optimizer,
            scaler,
            cfg,
            args.tokenizer,
            final_step,
            last_loss,
            append_eos,
            args.eos_loss_weight,
            clean_text,
        )


if __name__ == "__main__":
    main()
