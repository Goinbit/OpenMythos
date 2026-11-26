from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from open_mythos.config import MythosConfig
from open_mythos.primitives.quant import build_linear


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int, cfg: Optional[MythosConfig] = None):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
            cfg        -- when given, linears honor cfg.quant.mode (BitLinear
                          under "w158"); None keeps plain nn.Linear
        """
        super().__init__()

        def make_linear(in_features: int, out_features: int) -> nn.Linear:
            if cfg is not None:
                return build_linear(cfg, in_features, out_features)
            return nn.Linear(in_features, out_features, bias=False)

        self.gate = make_linear(dim, expert_dim)
        self.up = make_linear(dim, expert_dim)
        self.down = make_linear(expert_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _GroupedLinear(Function):
    """Grouped GEMM with an explicit backward for CPU/MPS/CUDA parity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        k, n = x.shape[1], weight.shape[1]
        pad_k = (-k) % 8
        pad_n = (-n) % 8
        x_padded = F.pad(x, (0, pad_k)) if pad_k else x
        weight_padded = F.pad(weight, (0, pad_k, 0, pad_n)) if pad_k or pad_n else weight
        ctx.save_for_backward(x_padded, weight_padded, offsets)
        ctx.k = k
        ctx.n = n
        ctx.pad_n = pad_n
        out = F.grouped_mm(
            x_padded, weight_padded.transpose(-2, -1), offs=offsets
        )
        return out[:, :n]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if ctx.pad_n:
            grad_output = F.pad(grad_output, (0, ctx.pad_n))
        grad_x = F.grouped_mm(grad_output, weight, offs=offsets)[:, : ctx.k].contiguous()
        grad_weight = F.grouped_mm(
            x.transpose(0, 1), grad_output, offs=offsets
        ).transpose(-2, -1)[:, : ctx.n, : ctx.k].contiguous()
        return grad_x, grad_weight, None


class GroupedExperts(nn.Module):
    """Packed SwiGLU experts evaluated as jagged grouped matrix multiplies."""

    def __init__(
        self,
        dim: int,
        expert_dim: int,
        n_experts: int,
        cfg: Optional[MythosConfig] = None,
    ):
        super().__init__()
        self.dim = dim
        self.expert_dim = expert_dim
        self.n_experts = n_experts
        self.quantized = cfg is not None and cfg.quant.mode == "w158"
        self.gate_weight = nn.Parameter(torch.empty(n_experts, expert_dim, dim))
        self.up_weight = nn.Parameter(torch.empty(n_experts, expert_dim, dim))
        self.down_weight = nn.Parameter(torch.empty(n_experts, dim, expert_dim))
        nn.init.normal_(self.gate_weight, std=0.02)
        nn.init.normal_(self.up_weight, std=0.02)
        nn.init.normal_(self.down_weight, std=0.02)

    def _effective_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return weight
        gamma = weight.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        ternary = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        dequantized = ternary * gamma
        return weight + (dequantized - weight).detach()

    def forward(
        self,
        sorted_x: torch.Tensor,
        offsets: torch.Tensor,
        topk_scores: torch.Tensor,
        token_ids: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        gate = _GroupedLinear.apply(
            sorted_x, self._effective_weight(self.gate_weight), offsets
        )
        up = _GroupedLinear.apply(
            sorted_x, self._effective_weight(self.up_weight), offsets
        )
        hidden = F.silu(gate) * up
        expert_out = _GroupedLinear.apply(
            hidden, self._effective_weight(self.down_weight), offsets
        )
        weighted = expert_out * topk_scores.unsqueeze(-1)
        out = torch.zeros(
            (n_tokens, self.dim), device=sorted_x.device, dtype=sorted_x.dtype
        )
        return out.index_add(0, token_ids, weighted)
