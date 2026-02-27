"""
GatedTrainer: drop-in training loop with hybrid gating.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List
import numpy as np

from .gates import GeometricGate, EntropyGate, HybridGate
from .loss import GatedLoss


class GatedTrainer:
    """
    Trains a PyTorch classifier with Hybrid Geometric-Entropy Gating.

    Usage:
        trainer = GatedTrainer(model, num_classes=4)
        trainer.fit(X_train, y_train, epochs=50)
        acc = trainer.evaluate(X_test, y_test)

    Args:
        model           : Any nn.Module that outputs logits of shape (N, C).
        num_classes     : Number of output classes.
        lr              : Learning rate.
        alpha           : Geometric gate steepness.
        beta            : Entropy gate steepness.
        warmup_epochs   : Epochs before gate is activated.
        device          : 'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 1e-3,
        alpha: float = 5.0,
        beta: float = 5.0,
        warmup_epochs: int = 5,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)

        # Gate components
        self.geo_gate = GeometricGate(alpha=alpha, learnable=False)
        self.ent_gate = EntropyGate(beta=beta, num_classes=num_classes)
        self.hybrid_gate = HybridGate(
            geometric_gate=self.geo_gate,
            entropy_gate=self.ent_gate,
            warmup_steps=0,  # handled at epoch level
        )
        self.gated_loss = GatedLoss(self.hybrid_gate)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.train_center: Optional[torch.Tensor] = None
        self.train_radius: Optional[float] = None
        self.history: List[Dict] = []

    def _compute_center(self, X: torch.Tensor):
        """Compute training distribution center and radius.
        Uses 75th percentile of distances as d0 — gives the gate
        more room and avoids cutting off valid training samples.
        """
        self.train_center = X.mean(dim=0).to(self.device)
        dists = torch.norm(X - self.train_center.cpu(), dim=1)
        self.train_radius = float(torch.quantile(dists, 0.75).item())
        self.geo_gate.log_d0.data = torch.tensor(self.train_radius).log().to(self.device)

    def _distances(self, X: torch.Tensor) -> torch.Tensor:
        """L2 distance from training center."""
        return torch.norm(X.to(self.device) - self.train_center, dim=1)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 64,
        verbose: bool = True,
    ):
        """Train the model with gated loss."""
        self._compute_center(X_train)
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        steps_per_epoch = len(loader)
        warmup_steps = self.warmup_epochs * steps_per_epoch
        self.hybrid_gate.warmup_steps = warmup_steps

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                dists = self._distances(X_batch)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.gated_loss(logits, y_batch, dists)
                loss.backward()
                self.optimizer.step()
                self.hybrid_gate.step()

                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            acc = correct / total
            avg_loss = epoch_loss / len(loader)
            self.history.append({"epoch": epoch + 1, "loss": avg_loss, "acc": acc})

            if verbose and (epoch + 1) % 10 == 0:
                gate_status = "WARMUP" if self.hybrid_gate.is_warming_up else "ACTIVE"
                print(f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | acc={acc:.3f} | gate={gate_status}")

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Return accuracy on given data."""
        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        logits = self.model(X)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.model.eval()
        logits = self.model(X.to(self.device))
        return torch.softmax(logits, dim=-1).cpu()

    @torch.no_grad()
    def gate_weights(self, X: torch.Tensor) -> torch.Tensor:
        """Return hybrid gate weights for visualization."""
        self.model.eval()
        X = X.to(self.device)
        dists = self._distances(X)
        logits = self.model(X)
        entropy = EntropyGate.entropy_from_logits(logits)
        # Temporarily disable warmup for evaluation
        orig = self.hybrid_gate.warmup_steps
        self.hybrid_gate.warmup_steps = 0
        weights = self.hybrid_gate(dists, entropy)
        self.hybrid_gate.warmup_steps = orig
        return weights.cpu()
