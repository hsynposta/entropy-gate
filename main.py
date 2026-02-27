"""
main.py — Hybrid Geometric-Entropy Gating
Full benchmark: multi-seed, multiple OOD shift levels, publication-quality figure.

Usage:
    python main.py              # full run (5 seeds × 4 shifts)
    python main.py --seeds 3    # faster
    python main.py --no-plot    # results only, no figure
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seeds",   type=int, default=5,    help="number of random seeds")
parser.add_argument("--epochs",  type=int, default=80,   help="training epochs")
parser.add_argument("--no-plot", action="store_true",    help="skip matplotlib figure")
args = parser.parse_args()

try:
    from entropy_gate import GatedTrainer
except ImportError:
    print("Run: pip install -e .")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
N_TRAIN     = 200          # samples per class
N_TEST      = 150
N_CLASSES   = 4
HIDDEN      = 64
LR          = 1e-3
BATCH       = 64
WARMUP      = 8
OOD_SHIFTS  = [0.0, 0.5, 1.0, 1.5, 2.0]   # covariate shift magnitudes
SEEDS       = list(range(args.seeds))

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_data(n, shift, scale, seed):
    rng = np.random.RandomState(seed)
    centers = np.array([[2,2],[-2,2],[-2,-2],[2,-2]], dtype=np.float32)
    X, y = [], []
    for c, center in enumerate(centers):
        pts = rng.randn(n, 2).astype(np.float32) * scale + center + shift
        X.append(pts); y.extend([c]*n)
    return torch.from_numpy(np.vstack(X)), torch.tensor(y, dtype=torch.long)

def build_mlp():
    return nn.Sequential(
        nn.Linear(2, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, N_CLASSES),
    )

def train_baseline(X_tr, y_tr, seed, epochs):
    torch.manual_seed(seed)
    model = build_mlp()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_gated(X_tr, y_tr, seed, epochs):
    torch.manual_seed(seed)
    model   = build_mlp()
    trainer = GatedTrainer(model, num_classes=N_CLASSES, lr=LR,
                           warmup_epochs=WARMUP, device="cpu")
    trainer.fit(X_tr, y_tr, epochs=epochs, batch_size=BATCH, verbose=False)
    return model, trainer

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return (model(X).argmax(1) == y).float().mean().item()

@torch.no_grad()
def mean_conf(model, X):
    model.eval()
    return torch.softmax(model(X), dim=1).max(1).values.mean().item()

# ── Experiment ───────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  Hybrid Geometric–Entropy Gating  |  Benchmark")
print("═"*60)
print(f"  Seeds: {args.seeds}  |  Epochs: {args.epochs}  |  Shifts: {OOD_SHIFTS}")
print("─"*60)

results = {s: {"base_acc":[], "gate_acc":[], "base_conf":[], "gate_conf":[]}
           for s in OOD_SHIFTS}

t0 = time.time()
for seed in SEEDS:
    # Training data (no shift)
    X_tr, y_tr = make_data(N_TRAIN, shift=0.0, scale=0.7, seed=seed*100)

    base_model          = train_baseline(X_tr, y_tr, seed, args.epochs)
    gate_model, trainer = train_gated(X_tr, y_tr, seed, args.epochs)

    # Evaluate at each OOD shift level
    for shift in OOD_SHIFTS:
        X_te, y_te = make_data(N_TEST, shift=shift, scale=1.2, seed=seed*100+1)
        results[shift]["base_acc"].append(accuracy(base_model, X_te, y_te))
        results[shift]["gate_acc"].append(accuracy(gate_model, X_te, y_te))
        results[shift]["base_conf"].append(mean_conf(base_model, X_te))
        results[shift]["gate_conf"].append(mean_conf(gate_model, X_te))

    sys.stdout.write(f"\r  Progress: seed {seed+1}/{args.seeds}  [{time.time()-t0:.1f}s]")
    sys.stdout.flush()

print("\n")

# ── Print results table ───────────────────────────────────────────────────────
def mean_std(lst): return np.mean(lst), np.std(lst)

print(f"  {'Shift':>6}  {'Baseline Acc':>14}  {'Gated Acc':>12}  {'Δ Acc':>8}  {'Base Conf':>10}  {'Gate Conf':>10}")
print("  " + "─"*72)

for shift in OOD_SHIFTS:
    r = results[shift]
    bm, bs = mean_std(r["base_acc"])
    gm, gs = mean_std(r["gate_acc"])
    bcm, _ = mean_std(r["base_conf"])
    gcm, _ = mean_std(r["gate_conf"])
    delta   = gm - bm
    arrow   = "▲" if delta > 0 else "▼"
    tag     = "  (ID)" if shift == 0 else ""
    print(f"  {shift:>6.1f}  "
          f"{bm*100:>8.1f}±{bs*100:.1f}%  "
          f"{gm*100:>6.1f}±{gs*100:.1f}%  "
          f"{arrow}{abs(delta)*100:>5.1f}pp  "
          f"{bcm*100:>8.1f}%  "
          f"{gcm*100:>8.1f}%"
          f"{tag}")

print("  " + "─"*72)
# Summary
ood_shifts = [s for s in OOD_SHIFTS if s > 0]
avg_delta = np.mean([np.mean(results[s]["gate_acc"]) - np.mean(results[s]["base_acc"])
                     for s in ood_shifts])
conf_reduction = np.mean([np.mean(results[s]["base_conf"]) - np.mean(results[s]["gate_conf"])
                          for s in ood_shifts])
print(f"\n  Average OOD accuracy gain : {avg_delta*100:+.1f} pp")
print(f"  Average confidence reduction on OOD: {conf_reduction*100:.1f} pp (better calibration)")
print(f"  Total time: {time.time()-t0:.1f}s")
print()

# ── Plot ─────────────────────────────────────────────────────────────────────
if not args.no_plot:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    shifts_arr  = np.array(OOD_SHIFTS)
    base_accs   = np.array([np.mean(results[s]["base_acc"]) for s in OOD_SHIFTS])
    gate_accs   = np.array([np.mean(results[s]["gate_acc"]) for s in OOD_SHIFTS])
    base_stds   = np.array([np.std(results[s]["base_acc"])  for s in OOD_SHIFTS])
    gate_stds   = np.array([np.std(results[s]["gate_acc"])  for s in OOD_SHIFTS])
    base_confs  = np.array([np.mean(results[s]["base_conf"]) for s in OOD_SHIFTS])
    gate_confs  = np.array([np.mean(results[s]["gate_conf"]) for s in OOD_SHIFTS])

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    DARK  = "#0f1117"
    GRID  = "#1e2130"
    BASE  = "#e74c3c"
    GATE  = "#2ecc71"
    GOLD  = "#f1c40f"
    TEXT  = "#ecf0f1"
    SUBTEXT = "#95a5a6"

    def style_ax(ax, title):
        ax.set_facecolor(GRID)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.spines[:].set_color("#2c3e50")
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)

    # ── Panel 1: Accuracy vs Shift ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(shifts_arr, base_accs - base_stds, base_accs + base_stds,
                     alpha=0.15, color=BASE)
    ax1.fill_between(shifts_arr, gate_accs - gate_stds, gate_accs + gate_stds,
                     alpha=0.15, color=GATE)
    ax1.plot(shifts_arr, base_accs, "o-", color=BASE, lw=2.5, ms=7,
             label="Baseline (CE)", zorder=3)
    ax1.plot(shifts_arr, gate_accs, "s-", color=GATE, lw=2.5, ms=7,
             label="Gated (Hybrid)", zorder=3)
    ax1.axvline(0, color=GOLD, lw=1, ls="--", alpha=0.5, label="ID boundary")
    ax1.set_xlabel("Covariate Shift Magnitude")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.3, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax1.legend(facecolor=DARK, edgecolor="#2c3e50", labelcolor=TEXT, fontsize=9)
    style_ax(ax1, "A  Accuracy vs OOD Shift")

    # ── Panel 2: Δ Accuracy bars ──
    ax2 = fig.add_subplot(gs[0, 2])
    deltas = gate_accs - base_accs
    colors = [GATE if d >= 0 else BASE for d in deltas]
    bars   = ax2.bar(shifts_arr, deltas * 100, color=colors, width=0.3,
                     edgecolor="#0f1117", linewidth=0.5, zorder=3)
    ax2.axhline(0, color=SUBTEXT, lw=0.8)
    ax2.set_xlabel("Shift Magnitude")
    ax2.set_ylabel("Accuracy Gain (pp)")
    for bar, val in zip(bars, deltas * 100):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + (0.3 if val >= 0 else -0.8),
                 f"{val:+.1f}", ha="center", va="bottom", color=TEXT, fontsize=8)
    style_ax(ax2, "B  Accuracy Gain (Gated − Baseline)")

    # ── Panel 3: Confidence vs Shift ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(shifts_arr, base_confs, "o-", color=BASE, lw=2.5, ms=7,
             label="Baseline confidence")
    ax3.plot(shifts_arr, gate_confs, "s-", color=GATE, lw=2.5, ms=7,
             label="Gated confidence")
    ax3.fill_between(shifts_arr, base_confs, gate_confs, alpha=0.08, color=GATE)
    ax3.axvline(0, color=GOLD, lw=1, ls="--", alpha=0.5)
    ax3.set_xlabel("Covariate Shift Magnitude")
    ax3.set_ylabel("Mean Max Confidence")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.legend(facecolor=DARK, edgecolor="#2c3e50", labelcolor=TEXT, fontsize=9)
    style_ax(ax3, "C  Confidence Calibration vs Shift  (lower = better on OOD)")

    # ── Panel 4: Summary card ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(GRID)
    ax4.axis("off")
    summary_lines = [
        ("Method", "Baseline", "Gated"),
        ("ID Acc",
         f"{base_accs[0]*100:.1f}%",
         f"{gate_accs[0]*100:.1f}%"),
        ("OOD Acc\n(avg)",
         f"{np.mean(base_accs[1:])*100:.1f}%",
         f"{np.mean(gate_accs[1:])*100:.1f}%"),
        ("OOD Conf\n(avg)",
         f"{np.mean(base_confs[1:])*100:.1f}%",
         f"{np.mean(gate_confs[1:])*100:.1f}%"),
        ("Avg Gain", "—", f"{avg_delta*100:+.1f} pp"),
    ]
    for i, row in enumerate(summary_lines):
        y = 0.88 - i * 0.18
        color0 = TEXT if i > 0 else GOLD
        ax4.text(0.05, y, row[0],  color=SUBTEXT, fontsize=9, va="center", transform=ax4.transAxes)
        ax4.text(0.48, y, row[1],  color=BASE,    fontsize=9, va="center", transform=ax4.transAxes, ha="center")
        ax4.text(0.82, y, row[2],  color=GATE,    fontsize=9, va="center", transform=ax4.transAxes, ha="center")
        if i == 0:
            ax4.plot([0.02, 0.98], [y - 0.08, y - 0.08],
                     color="#2c3e50", lw=0.8, transform=ax4.transAxes)
    ax4.set_title("D  Summary", color=TEXT, fontsize=10, fontweight="bold", pad=8)

    fig.suptitle(
        "Hybrid Geometric–Entropy Gating  ·  Aydin 2025\n"
        f"github.com/hsynposta/entropy-gate   |   {args.seeds} seeds × {len(OOD_SHIFTS)} shift levels",
        color=TEXT, fontsize=11, fontweight="bold", y=0.98
    )

    out = os.path.join(os.path.dirname(__file__), "benchmark.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure saved → {out}\n")
    plt.show()
