from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from omnitok.losses.alignment.base import BaseAlignmentLoss
from omnitok.registry import ALIGNMENT_REGISTRY


@ALIGNMENT_REGISTRY.register("vicreg")
class VICRegAlignmentLoss(BaseAlignmentLoss):
    """VICReg (Variance-Invariance-Covariance) style alignment loss.

    A powerful alignment strategy that explicitly prevents collapse and redundancy
    without requiring negative samples.

    Reference: VICReg: Variance-Invariance-Covariance Regularization for
    Self-Supervised Learning (ICLR 2022).

    Args:
        sim_weight: Weight for Invariance loss (MSE between features).
        var_weight: Weight for Variance loss (ensure std > 1).
        cov_weight: Weight for Covariance loss (decorrelation).
        epsilon: Small constant for numerical stability.
    """
    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        epsilon: float = 1e-4
    ) -> None:
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.epsilon = epsilon

    def compute(self, student_feat: Tensor, teacher_feat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute VICReg loss components."""
        # Normalize/flatten
        # student: (B, N, D) -> (B*N, D)
        # teacher: (B, N, D) -> (B*N, D)
        x = student_feat.flatten(0, 1)
        y = teacher_feat.flatten(0, 1)

        # 1. Invariance (Sim) loss: MSE between student and teacher
        sim_loss = F.mse_loss(x, y)

        # 2. Variance loss: Ensure each feature dimension has variance > 1
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)

        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        # 3. Covariance loss: Decorrelate different dimensions
        # (D, D) covariance matrix
        D = x.shape[1]
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)

        # Sum of non-diagonal elements squared
        def off_diagonal(mat):
            n, m = mat.shape
            assert n == m
            return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        cov_loss = off_diagonal(cov_x).pow_(2).sum() / D + off_diagonal(cov_y).pow_(2).sum() / D

        total_loss = (
            self.sim_weight * sim_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )

        return total_loss

@ALIGNMENT_REGISTRY.register("barlow_twins")
class BarlowTwinsAlignmentLoss(BaseAlignmentLoss):
    """Barlow Twins style alignment loss.

    Computes a cross-correlation matrix between student and teacher features
    and makes it close to the identity matrix.

    Reference: Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction (ICML 2021).
    """
    def __init__(self, lambd: float = 0.005) -> None:
        super().__init__()
        self.lambd = lambd

    def compute(self, student_feat: Tensor, teacher_feat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute Barlow Twins cross-correlation loss."""
        # (B*N, D)
        z1 = student_feat.flatten(0, 1)
        z2 = teacher_feat.flatten(0, 1)

        # Normalize along batch dimension
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        # (D, D) Cross-correlation matrix
        c = (z1.T @ z2) / z1.shape[0]

        # Identity matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Off-diagonal elements
        def off_diagonal(mat):
            n, m = mat.shape
            assert n == m
            return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag
        return loss
