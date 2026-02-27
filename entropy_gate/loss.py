"""
Gated loss function: L = (1/N) Σ w_hybrid(x_i) * ℓ(x_i, y_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .gates import HybridGate, EntropyGate


class GatedLoss(nn.Module):
    """
    Cross-entropy loss weighted by the hybrid gate.

    L_gated = (1/N) Σ_i  w_hybrid(x_i) * CE(f(x_i), y_i)

    The gate down-weights samples that are either:
      - Far from the training distribution (geometric gate), or
      - Predicted with high uncertainty (entropy gate).

    This forces the model to concentrate its capacity on the
    in-distribution manifold, improving OOD calibration.

    Args:
        gate            : HybridGate instance.
        reduction       : 'mean' (default) or 'sum'.
        label_smoothing : Optional label smoothing (passed to CE).
    """

    def __init__(
        self,
        gate: HybridGate,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gate = gate
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,          # (N, C)
        targets: torch.Tensor,         # (N,)
        distances: torch.Tensor,       # (N,)  — dist from train center
        center: Optional[torch.Tensor] = None,  # fallback if distances not precomputed
    ) -> torch.Tensor:
        """
        Returns scalar gated loss.
        """
        # Per-sample CE loss (unreduced)
        per_sample_ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # Prediction entropy for entropy gate
        entropy = EntropyGate.entropy_from_logits(logits.detach())

        # Hybrid gate weights
        weights = self.gate(distances, entropy)  # (N,)

        # Weighted loss
        weighted = weights * per_sample_ce

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        else:
            return weighted
