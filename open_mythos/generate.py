"""Autoregressive generation and static KV-cache preallocation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from open_mythos.kv_cache import StaticKVCache

if TYPE_CHECKING:
    from open_mythos.model import OpenMythos


def _preallocate_kv_cache(
    model: OpenMythos, input_ids: torch.Tensor, n_loops: int
) -> StaticKVCache:
    batch = input_ids.shape[0]
    device = input_ids.device
    device_type = device.type
    dtype = model.embed.weight.dtype
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
    cache = StaticKVCache(max_seq_len=model.cfg.max_seq_len)
    if model.cfg.attn_type == "mla":
        shapes = {
            "c_kv": (batch, model.cfg.max_seq_len, model.cfg.kv_lora_rank),
            "k_rope": (
                batch,
                model.cfg.max_seq_len,
                model.cfg.n_heads,
                model.cfg.qk_rope_head_dim,
            ),
        }
    else:
        head_dim = model.cfg.dim // model.cfg.n_heads
        shapes = {
            "k": (batch, model.cfg.max_seq_len, model.cfg.n_kv_heads, head_dim),
            "v": (batch, model.cfg.max_seq_len, model.cfg.n_kv_heads, head_dim),
        }
    layers = [f"prelude_{i}" for i in range(model.cfg.prelude_layers)]
    layers += [f"recurrent_loop_{i}" for i in range(n_loops)]
    layers += [f"coda_{i}" for i in range(model.cfg.coda_layers)]
    for layer in layers:
        for suffix, shape in shapes.items():
            cache.preallocate(f"{layer}:{suffix}", shape, dtype, device)
    return cache


@torch.no_grad()
def generate(
    model: OpenMythos,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    n_loops: int = 8,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """Generate tokens with a fully preallocated static KV cache."""
    kv_cache = _preallocate_kv_cache(
        model, input_ids, n_loops or model.cfg.max_loop_iters
    )
    prompt_len = input_ids.shape[1]
    for step in range(max_new_tokens):
        if step == 0:
            cur_ids = input_ids
            start_pos = 0
        else:
            cur_ids = input_ids[:, -1:]
            start_pos = prompt_len + step - 1
        if start_pos + cur_ids.shape[1] > model.cfg.max_seq_len:
            break
        logits = model(
            cur_ids,
            n_loops=n_loops,
            kv_cache=kv_cache,
            start_pos=start_pos,
            return_balance_loss=False,
        )
        if not isinstance(logits, torch.Tensor):
            raise TypeError("generate() requires logits without balance loss")
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    return input_ids
