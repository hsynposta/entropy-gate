"""
Core gate implementations from "Hybrid Geometric–Entropy Gating" (Aydin, 2025).

Three gate types:
  - GeometricGate  : distance-based soft boundary (heuristic or theory-derived)
  - EntropyGate    : prediction-entropy soft boundary
  - HybridGate     : element-wise product of both (the paper's main contribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class GeometricGate(nn.Module):
    """
    Soft range gate based on distance from training distribution center.

    Heuristic form (Eq. 1 in paper):
        w_range(x) = 1 / (1 + exp[ α * (d(x)/d0 - 1) ])

    Theory-derived form (finite-range field, Eq. 2):
        w_range(x) = 1 / (1 + (s(x)/s_c)^α )

    Args:
        d0          : Reference distance (training set radius). Learned if None.
        alpha       : Steepness of the sigmoid boundary.
        mode        : 'heuristic' (default) or 'theory'
        learnable   : If True, d0 and alpha are nn.Parameters.
    """

    def __init__(
        self,
        d0: float = 1.0,
        alpha: float = 5.0,
        mode: Literal["heuristic", "theory"] = "heuristic",
        learnable: bool = False,
    ):
        super().__init__()
        self.mode = mode

        if learnable:
            self.log_d0 = nn.Parameter(torch.tensor(d0).log())
            self.log_alpha = nn.Parameter(torch.tensor(alpha).log())
        else:
            self.register_buffer("log_d0", torch.tensor(d0).log())
            self.register_buffer("log_alpha", torch.tensor(alpha).log())

    @property
    def d0(self) -> torch.Tensor:
        return self.log_d0.exp()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Tensor of shape (N,) — L2 distance of each sample
                       from training distribution center.
        Returns:
            weights: Tensor of shape (N,) in [0, 1].
        """
        if self.mode == "heuristic":
            # w = sigmoid( -α * (d/d0 - 1) )
            return torch.sigmoid(-self.alpha * (distances / self.d0 - 1.0))
        else:
            # w = 1 / (1 + (d/d0)^α )    [finite-range field form]
            return 1.0 / (1.0 + (distances / self.d0) ** self.alpha)


class EntropyGate(nn.Module):
    """
    Soft gate based on prediction entropy of the model.

    High entropy  → model is uncertain → weight pushed toward 0 (ignore for training)
    Low entropy   → model is confident → weight close to 1

    Gate function (Eq. 3 in paper):
        w_ent(x) = 1 / (1 + exp[ β * (H(x) - H0) / ΔH ])

    where H(x) = -Σ p_k log p_k  (Shannon entropy of softmax output)

    Args:
        H0          : Entropy threshold (midpoint of gate). Default: log(C)/2
        beta        : Steepness. Higher = harder threshold.
        learnable   : If True, H0 and beta are nn.Parameters.
        num_classes : Used to set default H0 = log(C)/2 if H0 is None.
    """

    def __init__(
        self,
        H0: Optional[float] = None,
        beta: float = 5.0,
        learnable: bool = False,
        num_classes: int = 10,
    ):
        super().__init__()

        import math
        default_H0 = math.log(num_classes) / 2.0
        h0_val = H0 if H0 is not None else default_H0
        delta_H = math.log(num_classes)  # max possible entropy

        if learnable:
            self.log_beta = nn.Parameter(torch.tensor(beta).log())
            self.H0 = nn.Parameter(torch.tensor(h0_val))
        else:
            self.register_buffer("log_beta", torch.tensor(beta).log())
            self.register_buffer("H0", torch.tensor(h0_val))

        self.register_buffer("delta_H", torch.tensor(delta_H))

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    @staticmethod
    def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H(x) = -Σ p_k log p_k from raw logits."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)  # shape: (N,)

    @staticmethod
    def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy from probability vectors."""
        log_probs = (probs + 1e-12).log()
        return -(probs * log_probs).sum(dim=-1)

    def forward(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entropy: Tensor of shape (N,) — prediction entropy per sample.
        Returns:
            weights: Tensor of shape (N,) in [0, 1].
        """
        return torch.sigmoid(-self.beta * (entropy - self.H0) / self.delta_H)


class HybridGate(nn.Module):
    """
    Hybrid Geometric-Entropy Gate — the main contribution of the paper.

    w_hybrid(x) = w_range(x) * w_ent(x)

    A sample must be both:
      (1) geometrically close to the training distribution, AND
      (2) predicted with high confidence
    to receive full weight during training.

    Args:
        geometric_gate  : GeometricGate instance (or None to skip)
        entropy_gate    : EntropyGate instance (or None to skip)
        warmup_steps    : Number of steps during which gate is disabled (=1.0).
                          Matches the paper's warm-up protocol.
    """

    def __init__(
        self,
        geometric_gate: Optional[GeometricGate] = None,
        entropy_gate: Optional[EntropyGate] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.geometric_gate = geometric_gate or GeometricGate()
        self.entropy_gate = entropy_gate or EntropyGate()
        self.warmup_steps = warmup_steps
        self.register_buffer("_step", torch.tensor(0))

    def step(self):
        """Call once per training step to advance warmup counter."""
        self._step += 1

    @property
    def is_warming_up(self) -> bool:
        return self._step.item() < self.warmup_steps

    def forward(
        self,
        distances: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            distances : (N,) L2 distance from distribution center.
            entropy   : (N,) prediction entropy.
        Returns:
            weights   : (N,) hybrid gate values in [0, 1].
        """
        if self.is_warming_up:
            return torch.ones_like(distances)

        w_range = self.geometric_gate(distances)
        w_ent = self.entropy_gate(entropy)
        return w_range * w_ent
