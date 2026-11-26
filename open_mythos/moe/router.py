from typing import Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig


@runtime_checkable
class RouterProtocol(Protocol):
    """Structural contract for MoE token-to-expert routers."""

    n_experts: int
    topk: int

    def __call__(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class BiasRouter(nn.Module):
    """DeepSeek-V3 aux-loss-free bias router."""

    def __init__(self, dim: int, n_experts: int, topk: int):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.linear = nn.Linear(dim, n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(n_experts))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        # Normalize in fp32: 1e-9 underflows to zero in fp16, so a dtype-local
        # clamp still divides 0/0 when every selected softmax value underflows.
        normalized = topk_scores.float()
        denominator = normalized.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(normalized, 1.0 / self.topk)
        normalized = torch.where(
            denominator > 0,
            normalized / denominator.clamp(min=1e-9),
            uniform,
        )
        topk_scores = normalized.to(dtype=scores.dtype)
        return topk_idx, topk_scores, scores


def build_router(cfg: MythosConfig) -> RouterProtocol:
    """Construct the router module matching ``cfg``."""
    return BiasRouter(
        dim=cfg.dim,
        n_experts=cfg.moe.n_experts,
        topk=cfg.moe.n_experts_per_tok,
    )
