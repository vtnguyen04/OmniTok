"""Sparse Teacher Router — Dynamic teacher selection for multi-teacher distillation.

Instead of computing alignment loss against ALL teachers every step,
the router selects top-k teachers per sample based on encoder features.
This reduces compute (fewer teacher forwards), gradient conflict
(fewer conflicting alignment signals), and data requirements.

Inspired by Switch Transformer's load balancing mechanism.

Reference:
    - Switch Transformer (Fedus et al., 2021) — load balance loss
    - Mixture of Experts sparse routing literature
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..registry import TEACHER_ROUTER_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class TeacherRoutingResult:
    """Result of teacher routing decision.

    Attributes:
        selected_indices: (B, top_k) — indices of selected teachers per sample.
        gating_weights: (B, top_k) — normalized weights for selected teachers.
        load_balance_loss: Scalar — auxiliary loss to prevent router collapse.
        router_logits: (B, num_teachers) — raw logits for logging/analysis.
    """

    selected_indices: Tensor
    gating_weights: Tensor
    load_balance_loss: Tensor
    router_logits: Tensor
    router_probs: Tensor


@TEACHER_ROUTER_REGISTRY.register("sparse")
class SparseTeacherRouter(nn.Module):
    """Routes each sample to top-k teachers for sparse multi-teacher alignment.

    A lightweight gating network that takes global-averaged encoder features
    and produces a routing decision: which teachers to align with for each
    sample in the batch.

    Uses Switch Transformer-style load balancing loss to prevent the router
    from collapsing to always selecting the same teacher.

    Args:
        student_dim: Encoder feature dimension (e.g., 768).
        num_teachers: Total number of available teachers.
        top_k: Number of teachers to select per sample.
        temperature: Softmax temperature for gating weights.
            Lower = more peaked (confident), higher = more uniform.
        load_balance_weight: Weight for the auxiliary load balance loss.
            Set to 0.0 to disable.
        z_loss_weight: Weight for z-loss (ST-MoE) that penalizes large
            router logits to prevent confident collapse.
    """

    def __init__(
        self,
        student_dim: int,
        num_teachers: int,
        top_k: int = 1,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if top_k > num_teachers:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_teachers ({num_teachers})"
            )
        self.num_teachers = num_teachers
        self.top_k = top_k
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight
        self.z_loss_weight = z_loss_weight

        # Lightweight gate: pool → linear → routing logits
        self.gate = nn.Linear(student_dim, num_teachers, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

        logger.info(
            f"SparseTeacherRouter: {num_teachers} teachers, "
            f"top_k={top_k}, temp={temperature}, "
            f"lb_weight={load_balance_weight}, z_loss={z_loss_weight}"
        )

    def _compute_load_balance_loss(
        self,
        router_probs: Tensor,
        selected_indices: Tensor,
    ) -> Tensor:
        """Compute Switch Transformer load balance loss.

        Encourages uniform teacher utilization across the batch.

        L_balance = N * sum_i(f_i * P_i) where:
            f_i = fraction of samples routed to teacher i
            P_i = average routing probability for teacher i
            N = num_teachers

        At uniform distribution: L_balance = 1.0 (minimum).
        At full collapse: L_balance = N (maximum).

        Args:
            router_probs: Softmax probabilities (B, num_teachers).
            selected_indices: Selected teacher indices (B, top_k).

        Returns:
            Scalar load balance loss.
        """
        if self.load_balance_weight == 0.0:
            return torch.zeros(1, device=router_probs.device)

        # f_i: fraction of samples routed to each teacher
        one_hot = F.one_hot(selected_indices, self.num_teachers).float()
        # (B, top_k, num_teachers) → sum over top_k → (B, num_teachers) → mean over B
        fraction = one_hot.sum(dim=1).mean(dim=0)  # (num_teachers,)

        # P_i: average routing probability for each teacher
        avg_prob = router_probs.mean(dim=0)  # (num_teachers,)

        # Switch Transformer loss
        loss = self.num_teachers * (fraction * avg_prob).sum()

        return self.load_balance_weight * loss

    def _compute_z_loss(self, logits: Tensor) -> Tensor:
        """Z-loss from ST-MoE: penalizes large router logits.

        Prevents the router from becoming overconfident and collapsing
        to always selecting the same teacher. Complements load balance
        loss by acting on the raw logits rather than the routing decisions.

        L_z = mean(log(sum(exp(logits)))^2)

        Reference: ST-MoE (Zoph et al., 2022), Section 3.3

        Args:
            logits: Router logits (B, num_teachers).

        Returns:
            Scalar z-loss.
        """
        if self.z_loss_weight == 0.0:
            return torch.zeros(1, device=logits.device)

        log_z = torch.logsumexp(logits, dim=-1)  # (B,)
        z_loss = log_z.square().mean()
        return self.z_loss_weight * z_loss

    def forward(self, student_features: Tensor) -> TeacherRoutingResult:
        """Route each sample to top-k teachers.

        Args:
            student_features: Encoder features (B, N, D) — will be
                globally averaged to (B, D) for routing decision.

        Returns:
            TeacherRoutingResult with selected indices, weights, and losses.
        """
        # Global average pooling: (B, N, D) → (B, D)
        if student_features.ndim == 3:
            pooled = student_features.mean(dim=1)
        else:
            pooled = student_features

        # Router logits and probabilities
        logits = self.gate(pooled)  # (B, num_teachers)
        router_probs = F.softmax(logits / self.temperature, dim=-1)

        # Select top-k teachers per sample
        top_k_logits, selected_indices = torch.topk(
            logits, self.top_k, dim=-1
        )  # both (B, top_k)

        # Gating weights: re-normalize softmax over selected teachers only
        gating_weights = F.softmax(top_k_logits / self.temperature, dim=-1)

        # Anti-collapse losses
        lb_loss = self._compute_load_balance_loss(router_probs, selected_indices)
        z_loss = self._compute_z_loss(logits)
        total_aux_loss = lb_loss + z_loss

        return TeacherRoutingResult(
            selected_indices=selected_indices,
            gating_weights=gating_weights,
            load_balance_loss=total_aux_loss,
            router_logits=logits.detach(),
            router_probs=router_probs.detach(),
        )

    def get_teacher_usage(self, selected_indices: Tensor) -> Tensor:
        """Compute per-teacher usage fraction for logging.

        Args:
            selected_indices: (B, top_k) selected teacher indices.

        Returns:
            (num_teachers,) tensor with usage fraction per teacher.
        """
        one_hot = F.one_hot(selected_indices, self.num_teachers).float()
        usage = one_hot.sum(dim=1).mean(dim=0)  # (num_teachers,)
        return usage

    def get_routing_metrics(
        self,
        routing: TeacherRoutingResult,
        teacher_names: List[str],
    ) -> Dict[str, float]:
        """Compute comprehensive routing metrics for WandB logging.

        Args:
            routing: TeacherRoutingResult from forward().
            teacher_names: List of teacher names for labeling.

        Returns:
            Dict of metric_name → value for logging.
        """
        metrics: Dict[str, float] = {}

        # 1. Per-teacher usage fraction: what % of batch selects each teacher
        usage = self.get_teacher_usage(routing.selected_indices)
        for i, t_name in enumerate(teacher_names):
            metrics[f"router/usage_{t_name}"] = usage[i].item()

        # 2. Router entropy: measures diversity of routing decisions
        #    High entropy = diverse routing, low = collapsed
        probs = routing.router_probs  # (B, num_teachers)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        max_entropy = math.log(self.num_teachers)
        metrics["router/entropy"] = entropy.item()
        metrics["router/entropy_ratio"] = entropy.item() / max_entropy

        # 3. Max probability: average of the highest routing prob per sample
        #    Close to 1.0 = very confident (risk of collapse)
        #    Close to 1/N = uniform (healthy)
        max_prob = probs.max(dim=-1).values.mean()
        metrics["router/max_prob"] = max_prob.item()

        # 4. Per-teacher average gating weight (how much each teacher contributes)
        for i, t_name in enumerate(teacher_names):
            mask = (routing.selected_indices == i).float()  # (B, top_k)
            if mask.sum() > 0:
                avg_weight = (routing.gating_weights * mask).sum() / (mask.sum() + 1e-10)
                metrics[f"router/weight_{t_name}"] = avg_weight.item()
            else:
                metrics[f"router/weight_{t_name}"] = 0.0

        # 5. Collapse indicator: 1.0 = perfectly uniform, 0.0 = fully collapsed
        #    Uses coefficient of variation of usage
        usage_std = usage.std().item()
        usage_mean = usage.mean().item()
        if usage_mean > 0:
            cv = usage_std / usage_mean
            # CV = 0 when uniform, higher when imbalanced
            # Convert to 0-1 score: 1 = healthy, 0 = collapsed
            metrics["router/balance_score"] = max(0.0, 1.0 - cv)
        else:
            metrics["router/balance_score"] = 0.0

        return metrics

    def extra_repr(self) -> str:
        return (
            f"num_teachers={self.num_teachers}, top_k={self.top_k}, "
            f"temperature={self.temperature}, "
            f"load_balance_weight={self.load_balance_weight}, "
            f"z_loss_weight={self.z_loss_weight}"
        )
