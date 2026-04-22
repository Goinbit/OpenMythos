import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.primitives.expert import Expert


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # renorm

        # P3.2: vectorized routed-expert dispatch.
        # Flatten the (token, k_slot) pairs so every top-K assignment is one
        # entry in three parallel tensors. One expert at a time, we gather all
        # of its tokens in a single matmul instead of iterating per k-slot.
        N = flat.shape[0]
        K = self.topk
        flat_idx = topk_idx.reshape(-1)          # (N*K,)  expert id per assignment
        flat_weights = topk_scores.reshape(-1)   # (N*K,)  router weight per assignment
        flat_tokens = (
            torch.arange(N, device=flat.device)
            .unsqueeze(1)
            .expand(-1, K)
            .reshape(-1)
        )  # (N*K,)  token index per assignment

        out = torch.zeros_like(flat)
        # One sync to read bincount into Python — amortized across all experts,
        # far cheaper than per-expert .any() checks inside the inner loop.
        counts = torch.bincount(flat_idx, minlength=self.n_experts).tolist()
        for eid in range(self.n_experts):
            if counts[eid] == 0:
                continue
            mask = flat_idx == eid
            token_selector = flat_tokens[mask]          # (M,)
            weights = flat_weights[mask].unsqueeze(-1)  # (M, 1)
            expert_out = self.routed_experts[eid](flat[token_selector])  # (M, D)
            out.index_add_(0, token_selector, expert_out * weights)

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)
