"""Validation loop + periodic eval hook.

Validation uses a deterministic document-hash held-out bucket. Every FSDP rank
participates in forward and loss reduction; only rank 0 writes the log.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from training.data import FineWebEduDataset, _seed_worker


def build_validation_loader(
    encoding,
    rank: int,
    world_size: int,
    seq_len: int,
    subset: str = "sample-10BT",
    micro_batch: int = 4,
    num_workers: int = 2,
) -> DataLoader:
    """DataLoader over the deterministic held-out document bucket.

    FineWeb-Edu exposes only a train split, so ``FineWebEduDataset`` reserves
    a stable document-ID hash bucket for validation and excludes it from the
    training loader. Each rank still owns a disjoint shard of that bucket.

    Args:
        encoding    -- tokenizer exposing ``.encode``
        rank        -- current distributed rank (used for sharding)
        world_size  -- total distributed ranks
        seq_len     -- context length for validation examples
        subset      -- FineWeb-Edu config name (same default as training)
        micro_batch -- per-step validation batch size
        num_workers -- DataLoader workers; kept low since eval is infrequent

    Returns:
        DataLoader iterating over ``(input_ids, target_ids)`` pairs.
    """
    dataset = FineWebEduDataset(
        encoding, seq_len, subset, rank, world_size, split="validation"
    )
    return DataLoader(
        dataset,
        batch_size=micro_batch,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )


def compute_val_loss(
    model,
    loader: DataLoader,
    device: str | torch.device,
    vocab_size: int,
    num_batches: int = 10,
    ddp: bool = False,
) -> float:
    """Run eval on every rank and return the globally averaged CE loss.

    FSDP forward contains collectives, so non-master ranks must not skip eval.
    Loss sum and batch count are reduced once; only rank 0 logs the result.
    """
    was_training = model.training
    model.eval()
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            )
            stats[0] += loss.detach().to(dtype=torch.float64)
            stats[1] += 1
    if ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    return float((stats[0] / stats[1].clamp(min=1)).item())


class PeriodicEvalHook:
    """Run scheduled eval on every rank; log the reduced loss on rank 0 only."""

    def __init__(
        self,
        every_n_steps: int,
        val_loader: DataLoader,
        device: str | torch.device,
        vocab_size: int,
        num_batches: int = 10,
        logger_obj=None,
    ) -> None:
        self.every_n_steps = every_n_steps
        self.val_loader = val_loader
        self.device = device
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self.logger_obj = logger_obj

    def __call__(self, step: int, trainer) -> None:
        if step == 0 or step % self.every_n_steps != 0:
            return
        val_loss = compute_val_loss(
            trainer.model,
            self.val_loader,
            self.device,
            self.vocab_size,
            num_batches=self.num_batches,
            ddp=getattr(trainer, "ddp", False),
        )
        if not getattr(trainer, "master", True):
            return
        msg = f"[eval] step={step} val_loss={val_loss:.4f}"
        if self.logger_obj is not None:
            self.logger_obj.info(msg)
        else:
            print(msg)
