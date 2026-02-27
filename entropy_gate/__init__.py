"""
Entropy Gate — Hybrid Geometric-Entropy Gating for OOD-robust neural networks.

Based on: "Hybrid Geometric–Entropy Gating" (Aydin, 2025)

Quick start:
    from entropy_gate import GeometricGate, EntropyGate, HybridGate, GatedLoss
"""

from .gates import GeometricGate, EntropyGate, HybridGate
from .loss import GatedLoss
from .trainer import GatedTrainer

__version__ = "0.1.0"
__author__ = "Hiram Aydin"
__all__ = ["GeometricGate", "EntropyGate", "HybridGate", "GatedLoss", "GatedTrainer"]
