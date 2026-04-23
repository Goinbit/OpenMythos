"""Training-loop orchestration + checkpoint I/O for OpenMythos pretraining.

Extracted from training/3b_fine_web_edu.py at Stage 4.5. The ``Trainer``
class wraps the monolithic main-loop body so the entry point can stay
thin and an eval hook can be injected without mutating the loop itself.
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

from training.schedule import get_lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in `ckpt_dir` sorted oldest → newest.

    Relies on the zero-padded `step_{0000000}.pt` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both `keep_last` pruning and resume-latest on startup, since both pick
    the last element of this list.

    Args:
        ckpt_dir -- directory to scan; missing directory returns []

    Returns:
        Sorted list of absolute paths to matching checkpoint files.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Under FSDP both states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. The temp-file + os.replace write means
    a kill mid-save leaves the previous checkpoint intact instead of a
    truncated .pt file. Non-master ranks participate in the FSDP gather
    (otherwise the collective would hang) but exit before touching disk.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optimizer   -- the optimizer whose state should round-trip with the model
        step        -- global step number; encoded zero-padded into the filename
        cfg         -- model config object; saved so downstream eval can
                       reconstruct the model without re-importing the variant
        vocab_size  -- tokenizer vocab size at train time; saved for sanity-check
                       on load against a (possibly updated) tokenizer
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- whether this rank writes to disk (rank 0 only)
        keep_last   -- number of most-recent checkpoints to retain; older ones
                       are unlinked after a successful write

    Returns:
        None. Writes to disk as a side effect on master rank.
    """
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    for name, tensor in model_state.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite parameter {name!r} at checkpoint save — refusing to write"
            )

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved → {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool, cfg=None) -> int:
    """
    Restore model + optimizer from disk, returning the step to resume at.

    Every rank reads the file (`rank0_only=False` on load) so FSDP has access
    to the full state on each rank — the complement to the `rank0_only=True`
    save path. Must mirror save's single-context pattern; splitting the model
    and optimizer loads across two `state_dict_type` blocks has historically
    produced optimizer state bound to the wrong shard shapes.

    `weights_only=False` is required because the checkpoint contains the
    pickled `cfg` dataclass — flip to `weights_only=True` only if you
    separate config out.

    Args:
        model     -- same FSDP-wrapped or raw model used during save
        optimizer -- freshly constructed optimizer to be filled in-place
        path      -- absolute path to a `step_{N:07d}.pt` file produced by
                     `save_checkpoint`
        ddp       -- whether the model is FSDP-wrapped; must match the save run
        cfg       -- current run's model config; when provided, the checkpoint's
                     stored `vocab_size` is compared against `cfg.vocab_size`
                     and a mismatch raises ValueError. Defended here so a
                     tokenizer swap between runs fails loud instead of loading
                     weights with the wrong embedding table shape.

    Returns:
        The step number the checkpoint was taken at; the caller advances the
        training loop from this value.

    Raises:
        ValueError if `cfg` is provided and the checkpoint's vocab_size
        disagrees with `cfg.vocab_size`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if cfg is not None:
        ckpt_vocab = ckpt.get("vocab_size")
        if ckpt_vocab is not None and ckpt_vocab != cfg.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size ({ckpt_vocab}) does not match "
                f"current cfg.vocab_size ({cfg.vocab_size}); tokenizer may have changed"
            )

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Encapsulates the FSDP training loop + checkpoint scheduling.

    Stage 4.5. Moved from training/3b_fine_web_edu.py::main() body. The class
    owns the stateful handles (model, optimizer, dataloader, AMP context) and
    exposes ``run``, ``save``, ``load`` as the public surface. An optional
    ``eval_hook`` callable is invoked as ``eval_hook(step, self)`` at each
    optimizer step so validation can be wired in without touching the loop.
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        dataloader,
        *,
        device: str,
        local_rank: int,
        rank: int,
        world_size: int,
        ddp: bool,
        master: bool,
        amp_ctx,
        vocab_size: int,
        ckpt_dir: str,
        total_steps: int,
        log_every: int,
        ckpt_every: int,
        grad_accum_steps: int,
        warmup_steps: int,
        max_lr: float,
        min_lr_ratio: float = 0.1,
        grad_clip: float = 1.0,
        global_batch_tok: int | None = None,
        eval_hook: Optional[Callable[[int, "Trainer"], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.ddp = ddp
        self.master = master
        self.amp_ctx = amp_ctx
        self.vocab_size = vocab_size
        self.ckpt_dir = ckpt_dir
        self.total_steps = total_steps
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.grad_accum_steps = grad_accum_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.global_batch_tok = global_batch_tok
        self.eval_hook = eval_hook

    # -- checkpoint helpers ------------------------------------------------

    def save(self, step: int) -> None:
        """Persist current model + optimizer state; master-only disk write."""
        save_checkpoint(
            self.model,
            self.optimizer,
            step,
            self.cfg,
            self.vocab_size,
            self.ckpt_dir,
            self.ddp,
            self.master,
        )

    def load(self, path: str) -> int:
        """Restore from ``path`` into this trainer's model/optimizer in-place."""
        return load_checkpoint(
            self.model, self.optimizer, path, self.ddp, cfg=self.cfg
        )

    # -- main loop ---------------------------------------------------------

    def run(self, start_step: int = 0) -> None:
        """Run the training loop from ``start_step`` to ``self.total_steps``.

        Body copied from 3b_fine_web_edu.py::main(). References to local
        variables are swapped for ``self.xxx``; the optional eval hook is
        invoked after each optimizer step so held-out loss can be computed
        at a fixed cadence without perturbing grad-accum timing.
        """
        if self.master:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model.train()
        data_iter = iter(self.dataloader)
        t0 = time.perf_counter()
        step = start_step

        min_lr = self.max_lr * self.min_lr_ratio
        target_device = (
            self.device if not self.ddp else f"cuda:{self.local_rank}"
        )

        while step < self.total_steps:
            cur_lr = get_lr(
                step,
                self.warmup_steps,
                self.total_steps,
                self.max_lr,
                min_lr,
            )
            for g in self.optimizer.param_groups:
                g["lr"] = cur_lr

            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for micro_step in range(self.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    x, y = next(data_iter)

                x = x.to(target_device, non_blocking=True)
                y = y.to(target_device, non_blocking=True)

                sync = (
                    nullcontext()
                    if (not self.ddp or micro_step == self.grad_accum_steps - 1)
                    else self.model.no_sync()
                )
                with sync, self.amp_ctx:
                    # TODO(P3.8): when USE_GRAD_CHECKPOINT is True, wrap the
                    # RecurrentBlock forward (not the whole model) with
                    # torch.utils.checkpoint.checkpoint(..., use_reentrant=False).
                    # Left as TODO until FSDP+activation-checkpoint interaction
                    # is validated.
                    logits = self.model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, self.vocab_size), y.view(-1)
                    )
                    loss = loss / self.grad_accum_steps

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"Non-finite loss at step {step}: {loss_value}"
                    )

                loss.backward()
                loss_accum += loss_value

            # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
            # against each rank's local norm and miss the cross-shard gather.
            # FSDP.clip_grad_norm_ computes the true global norm and returns it.
            if self.ddp:
                grad_norm = self.model.clip_grad_norm_(self.grad_clip)
            else:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
            self.optimizer.step()
            step += 1

            if self.master and step % self.log_every == 0:
                dt = time.perf_counter() - t0
                if self.global_batch_tok is not None:
                    tok_per_sec = self.global_batch_tok * self.log_every / dt
                    tokens_seen = step * self.global_batch_tok
                    logger.info(
                        f"step {step:6d}/{self.total_steps} | loss {loss_accum:.4f} "
                        f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                        f"| {tok_per_sec / 1e6:.2f}M tok/s "
                        f"| {tokens_seen / 1e9:.1f}B tokens seen"
                    )
                else:
                    logger.info(
                        f"step {step:6d}/{self.total_steps} | loss {loss_accum:.4f} "
                        f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e}"
                    )
                t0 = time.perf_counter()

            if step % self.ckpt_every == 0:
                self.save(step)

            if self.eval_hook is not None:
                self.eval_hook(step, self)

        # Final checkpoint — total_steps may not be divisible by ckpt_every,
        # so without this the tail of the run is lost if the schedule doesn't
        # align.
        if step > start_step and step % self.ckpt_every != 0:
            self.save(step)

        if self.ddp:
            # Barrier so no rank exits while another is still finishing its
            # checkpoint gather — avoids NCCL "process group destroyed" noise.
            dist.barrier()
            dist.destroy_process_group()

        if self.master:
            logger.success("Training complete.")
