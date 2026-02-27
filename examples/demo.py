"""
demo.py — Reproduces the core experiment from "Hybrid Geometric-Entropy Gating" (Aydin, 2025).

What this does:
  1. Generates a 2D 4-class dataset (concentric Gaussian blobs)
  2. Trains a baseline MLP (standard cross-entropy)
  3. Trains a gated MLP (HybridGate loss)
  4. Applies a covariate shift (OOD test set)
  5. Compares accuracy, confidence, and gate visualization

Run:
    python examples/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import TensorDataset, DataLoader

from entropy_gate import GeometricGate, EntropyGate, HybridGate, GatedLoss, GatedTrainer

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Data generation ────────────────────────────────────────────────────────
def make_blobs_2d(n=300, shift=0.0, scale=1.0, seed=42):
    """4-class 2D Gaussian blobs, optionally shifted/scaled for OOD."""
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=np.float32)
    X, y = [], []
    for c_idx, center in enumerate(centers):
        pts = rng.randn(n, 2).astype(np.float32) * scale + center + shift
        X.append(pts)
        y.extend([c_idx] * n)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

X_train, y_train = make_blobs_2d(n=200, shift=0.0, scale=0.7, seed=0)
X_id_test, y_id_test = make_blobs_2d(n=100, shift=0.0, scale=0.7, seed=1)
X_ood_test, y_ood_test = make_blobs_2d(n=100, shift=1.5, scale=1.4, seed=2)  # shifted + wider

print(f"Train: {X_train.shape}  |  ID test: {X_id_test.shape}  |  OOD test: {X_ood_test.shape}")

# ── 2. Model architecture ─────────────────────────────────────────────────────
def build_mlp(hidden=64):
    return nn.Sequential(
        nn.Linear(2, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 4),
    )

# ── 3. Baseline training (standard CE) ───────────────────────────────────────
print("\n── Training Baseline (standard CE) ──")
baseline = build_mlp()
opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()
for epoch in range(100):
    baseline.train()
    logits = baseline(X_train)
    loss = ce_loss(logits, y_train)
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch + 1) % 25 == 0:
        with torch.no_grad():
            baseline.eval()
            acc = (baseline(X_train).argmax(1) == y_train).float().mean()
        print(f"  Epoch {epoch+1:3d} | loss={loss.item():.4f} | train_acc={acc:.3f}")

# ── 4. Gated training ─────────────────────────────────────────────────────────
print("\n── Training Gated Model (HybridGate) ──")
gated_model = build_mlp()
trainer = GatedTrainer(
    gated_model,
    num_classes=4,
    lr=1e-3,
    alpha=5.0,
    beta=5.0,
    warmup_epochs=10,
)
trainer.fit(X_train, y_train, epochs=100, verbose=True)

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def mean_confidence(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).max(dim=1).values.mean().item()

print("\n── Results ──────────────────────────────────────────────")
print(f"{'':30s}  {'Baseline':>10}  {'Gated':>10}")
print("-" * 54)
print(f"{'ID Accuracy':30s}  {accuracy(baseline, X_id_test, y_id_test):10.3f}  {accuracy(gated_model, X_id_test, y_id_test):10.3f}")
print(f"{'OOD Accuracy':30s}  {accuracy(baseline, X_ood_test, y_ood_test):10.3f}  {accuracy(gated_model, X_ood_test, y_ood_test):10.3f}")
print(f"{'ID Mean Confidence':30s}  {mean_confidence(baseline, X_id_test):10.3f}  {mean_confidence(gated_model, X_id_test):10.3f}")
print(f"{'OOD Mean Confidence':30s}  {mean_confidence(baseline, X_ood_test):10.3f}  {mean_confidence(gated_model, X_ood_test):10.3f}")

# ── 6. Visualization ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle("Hybrid Geometric–Entropy Gating (Aydin, 2025)", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
MARKERS = ["o", "s", "^", "D"]
xx = np.linspace(-6, 6, 300)
yy = np.linspace(-6, 6, 300)
XX, YY = np.meshgrid(xx, yy)
grid_pts = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)

def decision_map(model, grid):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(grid), dim=1)
        return probs.max(dim=1).values.numpy().reshape(XX.shape), \
               probs.argmax(dim=1).numpy().reshape(XX.shape)

def scatter_dataset(ax, X, y, alpha=0.7, size=20, label_suffix=""):
    for c in range(4):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[c], marker=MARKERS[c],
                   s=size, alpha=alpha, label=f"Class {c}{label_suffix}", edgecolors="none")

# Panel A: Training data
ax0 = fig.add_subplot(gs[0])
scatter_dataset(ax0, X_train.numpy(), y_train.numpy())
ax0.set_title("A  Training Data", fontweight="bold")
ax0.set_xlim(-6, 6); ax0.set_ylim(-6, 6)
ax0.legend(fontsize=7, loc="upper right"); ax0.set_aspect("equal")

# Panel B: Baseline decision boundary + OOD
ax1 = fig.add_subplot(gs[1])
conf_b, cls_b = decision_map(baseline, grid_pts)
ax1.contourf(XX, YY, cls_b, alpha=0.12, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
             colors=COLORS)
ax1.contour(XX, YY, conf_b, levels=[0.5, 0.75, 0.9], colors="gray",
            linewidths=0.7, linestyles="--")
scatter_dataset(ax1, X_ood_test.numpy(), y_ood_test.numpy(), alpha=0.85, size=25, label_suffix=" (OOD)")
ax1.set_title(f"B  Baseline  OOD acc={accuracy(baseline, X_ood_test, y_ood_test):.2%}", fontweight="bold")
ax1.set_xlim(-6, 6); ax1.set_ylim(-6, 6); ax1.set_aspect("equal")

# Panel C: Gated decision boundary + OOD
ax2 = fig.add_subplot(gs[2])
conf_g, cls_g = decision_map(gated_model, grid_pts)
ax2.contourf(XX, YY, cls_g, alpha=0.12, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
             colors=COLORS)
ax2.contour(XX, YY, conf_g, levels=[0.5, 0.75, 0.9], colors="gray",
            linewidths=0.7, linestyles="--")
scatter_dataset(ax2, X_ood_test.numpy(), y_ood_test.numpy(), alpha=0.85, size=25, label_suffix=" (OOD)")
ax2.set_title(f"C  Gated Model  OOD acc={accuracy(gated_model, X_ood_test, y_ood_test):.2%}", fontweight="bold")
ax2.set_xlim(-6, 6); ax2.set_ylim(-6, 6); ax2.set_aspect("equal")

# Panel D: Gate weight map
ax3 = fig.add_subplot(gs[3])
gate_weights = trainer.gate_weights(grid_pts).numpy().reshape(XX.shape)
im = ax3.contourf(XX, YY, gate_weights, levels=50, cmap="RdYlGn", vmin=0, vmax=1)
plt.colorbar(im, ax=ax3, shrink=0.8)
# Overlay train center
cx, cy = trainer.train_center.cpu().numpy()
ax3.scatter([cx], [cy], marker="x", s=100, c="black", linewidths=2, zorder=5)
scatter_dataset(ax3, X_train.numpy(), y_train.numpy(), alpha=0.4, size=12)
ax3.set_title("D  Hybrid Gate w(x)\n(green=high, red=low)", fontweight="bold")
ax3.set_xlim(-6, 6); ax3.set_ylim(-6, 6); ax3.set_aspect("equal")

out_path = os.path.join(os.path.dirname(__file__), "results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
plt.show()
