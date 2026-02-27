"""
main.py — Hybrid Geometric-Entropy Gating
Real-data benchmark: sklearn Digits (64-dim, 10 classes)

Key scenario: training data is CONTAMINATED with noisy samples from a shifted domain.
- Baseline  : trains on everything equally → memorizes noise → weaker generalization
- Gated     : automatically downweights noisy training samples → better generalization

OOD eval: clean test digits + noise-corrupted test digits.

Usage:
    python main.py
    python main.py --seeds 3 --no-plot
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",       type=int, default=5)
parser.add_argument("--epochs",      type=int, default=100)
parser.add_argument("--noise-ratio", type=float, default=0.35,
                    help="fraction of TRAINING samples that are noise-contaminated")
parser.add_argument("--no-plot",     action="store_true")
args = parser.parse_args()

try:
    from entropy_gate import GatedTrainer
except ImportError:
    print("Run: pip install -e .")
    sys.exit(1)

try:
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Run: pip3 install scikit-learn")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
N_CLASSES    = 10
HIDDEN       = 128
LR           = 5e-4
BATCH        = 64
WARMUP       = 15
NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]   # test-time OOD noise
SEEDS        = list(range(args.seeds))
NOISE_RATIO  = args.noise_ratio              # fraction of contaminated train samples

# ── Data ──────────────────────────────────────────────────────────────────────
digits = load_digits()
X_all  = digits.data.astype(np.float32)
y_all  = digits.target.astype(np.int64)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_mlp():
    return nn.Sequential(
        nn.Linear(64, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, N_CLASSES),
    )

def contaminate(X, y, noise_ratio, noise_sigma, seed):
    """
    Simulate domain shift in training data:
    - Keep (1 - noise_ratio) of samples clean
    - Replace noise_ratio of samples with heavily noised versions + shuffled labels
    The gate should learn to downweight these automatically.
    """
    rng   = np.random.RandomState(seed)
    n     = len(X)
    n_bad = int(n * noise_ratio)
    idx   = rng.permutation(n)
    bad   = idx[:n_bad]

    X_c, y_c = X.clone(), y.clone()
    noise     = torch.from_numpy(rng.randn(n_bad, X.shape[1]).astype(np.float32)) * noise_sigma
    X_c[bad]  = X_c[bad] + noise
    y_c[bad]  = torch.from_numpy(rng.randint(0, N_CLASSES, size=n_bad).astype(np.int64))
    return X_c, y_c

# ── Training ──────────────────────────────────────────────────────────────────
def train_baseline(X_tr, y_tr, seed, epochs):
    torch.manual_seed(seed)
    model   = build_mlp()
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                     generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_gated(X_tr, y_tr, seed, epochs):
    torch.manual_seed(seed)
    model   = build_mlp()
    trainer = GatedTrainer(
        model, num_classes=N_CLASSES, lr=LR,
        alpha=4.0, beta=4.0,
        warmup_epochs=WARMUP, device="cpu",
    )
    trainer.fit(X_tr, y_tr, epochs=epochs, batch_size=BATCH, verbose=False)
    return model, trainer

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return (model(X).argmax(1) == y).float().mean().item()

@torch.no_grad()
def ece(model, X, y, n_bins=10):
    model.eval()
    probs       = torch.softmax(model(X), dim=1)
    confs, pred = probs.max(1)
    accs        = (pred == y).float()
    bins        = torch.linspace(0, 1, n_bins + 1)
    e           = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum() > 0:
            e += m.sum().item() * abs(accs[m].mean() - confs[m].mean()).item()
    return e / len(y)

# ── Experiment ────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  Hybrid Geometric–Entropy Gating  |  Contaminated Training")
print("═"*65)
print(f"  Dataset      : Digits (64-dim, 10 classes, {len(X_all)} samples)")
print(f"  Train noise  : {NOISE_RATIO:.0%} of training samples contaminated")
print(f"  Test OOD     : Gaussian noise σ = {NOISE_LEVELS}")
print(f"  Seeds        : {args.seeds}  |  Epochs: {args.epochs}")
print("─"*65)

results = {s: {"base_acc":[], "gate_acc":[],
               "base_ece":[], "gate_ece":[]}
           for s in NOISE_LEVELS}

t0 = time.time()
for seed in SEEDS:
    # Split
    X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
        X_all, y_all, test_size=0.25, random_state=seed, stratify=y_all
    )
    scaler = StandardScaler().fit(X_tr_np)
    X_tr   = torch.from_numpy(scaler.transform(X_tr_np))
    X_te   = torch.from_numpy(scaler.transform(X_te_np))
    y_tr   = torch.from_numpy(y_tr_np)
    y_te   = torch.from_numpy(y_te_np)

    # Contaminate training data
    X_tr_c, y_tr_c = contaminate(X_tr, y_tr, NOISE_RATIO, noise_sigma=2.5, seed=seed)

    # Both models see same contaminated training data
    base_model          = train_baseline(X_tr_c, y_tr_c, seed, args.epochs)
    gate_model, trainer = train_gated(X_tr_c, y_tr_c, seed, args.epochs)

    rng = np.random.RandomState(seed + 777)
    for noise in NOISE_LEVELS:
        if noise == 0.0:
            X_ev = X_te
        else:
            X_ev = X_te + torch.from_numpy(
                rng.randn(*X_te.shape).astype(np.float32) * noise
            )
        results[noise]["base_acc"].append(accuracy(base_model, X_ev, y_te))
        results[noise]["gate_acc"].append(accuracy(gate_model, X_ev, y_te))
        results[noise]["base_ece"].append(ece(base_model, X_ev, y_te))
        results[noise]["gate_ece"].append(ece(gate_model, X_ev, y_te))

    sys.stdout.write(f"\r  Seed {seed+1}/{args.seeds}  [{time.time()-t0:.1f}s]")
    sys.stdout.flush()

print("\n")

# ── Table ─────────────────────────────────────────────────────────────────────
def ms(lst): return np.mean(lst), np.std(lst)

print(f"  {'Noise σ':>8}  {'Base Acc':>12}  {'Gate Acc':>12}  {'Δ Acc':>8}  {'Base ECE':>10}  {'Gate ECE':>10}")
print("  " + "─"*72)
for noise in NOISE_LEVELS:
    r       = results[noise]
    bm, bs  = ms(r["base_acc"])
    gm, gs  = ms(r["gate_acc"])
    bem,_   = ms(r["base_ece"])
    gem,_   = ms(r["gate_ece"])
    delta   = gm - bm
    arrow   = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "─")
    tag     = "  ← clean test" if noise == 0 else ""
    print(f"  {noise:>8.1f}  "
          f"{bm*100:>7.1f}±{bs*100:.1f}%  "
          f"{gm*100:>7.1f}±{gs*100:.1f}%  "
          f"{arrow}{abs(delta)*100:>5.1f}pp  "
          f"{bem*100:>8.2f}%  "
          f"{gem*100:>8.2f}%"
          f"{tag}")

ood = [s for s in NOISE_LEVELS if s > 0]
avg_acc = np.mean([np.mean(results[s]["gate_acc"]) - np.mean(results[s]["base_acc"]) for s in ood])
avg_ece = np.mean([np.mean(results[s]["base_ece"]) - np.mean(results[s]["gate_ece"]) for s in ood])
print("  " + "─"*72)
print(f"\n  Average OOD accuracy gain    : {avg_acc*100:+.2f} pp")
print(f"  Average ECE improvement      : {avg_ece*100:+.2f} pp  (positive = better)")
print(f"  Total time: {time.time()-t0:.1f}s\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
if not args.no_plot:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    na   = np.array(NOISE_LEVELS)
    bacc = np.array([np.mean(results[s]["base_acc"]) for s in NOISE_LEVELS])
    gacc = np.array([np.mean(results[s]["gate_acc"]) for s in NOISE_LEVELS])
    bstd = np.array([np.std(results[s]["base_acc"])  for s in NOISE_LEVELS])
    gstd = np.array([np.std(results[s]["gate_acc"])  for s in NOISE_LEVELS])
    bece = np.array([np.mean(results[s]["base_ece"]) for s in NOISE_LEVELS])
    gece = np.array([np.mean(results[s]["gate_ece"]) for s in NOISE_LEVELS])

    DARK="#0f1117"; GRID="#1e2130"; BASE="#e74c3c"
    GATE="#2ecc71"; GOLD="#f1c40f"; TEXT="#ecf0f1"; SUB="#95a5a6"

    def style(ax, title):
        ax.set_facecolor(GRID); ax.tick_params(colors=SUB, labelsize=9)
        ax.spines[:].set_color("#2c3e50")
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.xaxis.label.set_color(SUB); ax.yaxis.label.set_color(SUB)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A — accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(na, bacc-bstd, bacc+bstd, alpha=0.15, color=BASE)
    ax1.fill_between(na, gacc-gstd, gacc+gstd, alpha=0.15, color=GATE)
    ax1.plot(na, bacc, "o-", color=BASE, lw=2.5, ms=7, label="Baseline (sees all noise equally)")
    ax1.plot(na, gacc, "s-", color=GATE, lw=2.5, ms=7, label="Gated (downweights noisy train samples)")
    ax1.axvline(0, color=GOLD, lw=1, ls="--", alpha=0.5, label="Clean test")
    ax1.set_xlabel("Test Noise Level (σ)"); ax1.set_ylabel("Accuracy")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax1.legend(facecolor=DARK, edgecolor="#2c3e50", labelcolor=TEXT, fontsize=9)
    style(ax1, f"A  Accuracy  |  Training set: {NOISE_RATIO:.0%} contaminated")

    # B — delta bars
    ax2 = fig.add_subplot(gs[0, 2])
    deltas = (gacc - bacc) * 100
    cols   = [GATE if d >= 0 else BASE for d in deltas]
    bars   = ax2.bar(na, deltas, color=cols, width=0.3, edgecolor=DARK, zorder=3)
    ax2.axhline(0, color=SUB, lw=0.8)
    ax2.set_xlabel("Test Noise (σ)"); ax2.set_ylabel("Accuracy Gain (pp)")
    for bar, val in zip(bars, deltas):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 val+(0.2 if val>=0 else -0.5),
                 f"{val:+.1f}", ha="center", fontsize=8, color=TEXT)
    style(ax2, "B  Accuracy Gain")

    # C — ECE
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(na, bece*100, "o-", color=BASE, lw=2.5, ms=7, label="Baseline ECE")
    ax3.plot(na, gece*100, "s-", color=GATE, lw=2.5, ms=7, label="Gated ECE")
    ax3.fill_between(na, bece*100, gece*100, alpha=0.1,
                     color=GATE if avg_ece > 0 else BASE)
    ax3.set_xlabel("Test Noise Level (σ)"); ax3.set_ylabel("ECE (%) — lower is better")
    ax3.legend(facecolor=DARK, edgecolor="#2c3e50", labelcolor=TEXT, fontsize=9)
    style(ax3, "C  Calibration Error (ECE) vs Noise")

    # D — summary
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(GRID); ax4.axis("off")
    rows = [
        ("Metric",          "Baseline",                              "Gated"),
        ("Clean Acc",       f"{bacc[0]*100:.1f}%",                  f"{gacc[0]*100:.1f}%"),
        ("OOD Acc (avg)",   f"{np.mean(bacc[1:])*100:.1f}%",        f"{np.mean(gacc[1:])*100:.1f}%"),
        ("OOD ECE (avg)",   f"{np.mean(bece[1:])*100:.2f}%",        f"{np.mean(gece[1:])*100:.2f}%"),
        ("Train contamination", f"{NOISE_RATIO:.0%} noise",         "same data"),
        ("Avg gain",        "—",                                     f"{avg_acc*100:+.2f} pp"),
    ]
    for i, row in enumerate(rows):
        yp = 0.92 - i*0.15
        ax4.text(0.03, yp, row[0], color=SUB,  fontsize=8.5, va="center", transform=ax4.transAxes)
        ax4.text(0.52, yp, row[1], color=BASE, fontsize=8.5, va="center", transform=ax4.transAxes, ha="center")
        ax4.text(0.87, yp, row[2], color=GATE, fontsize=8.5, va="center", transform=ax4.transAxes, ha="center")
        if i == 0:
            ax4.plot([0.02,0.98],[yp-0.06,yp-0.06], color="#2c3e50", lw=0.8, transform=ax4.transAxes)
    ax4.set_title("D  Summary", color=TEXT, fontsize=10, fontweight="bold", pad=8)

    fig.suptitle(
        "Hybrid Geometric–Entropy Gating  ·  Aydin 2025\n"
        f"github.com/hsynposta/entropy-gate   |   Digits  |  {NOISE_RATIO:.0%} train contamination  |  {args.seeds} seeds",
        color=TEXT, fontsize=11, fontweight="bold", y=0.98
    )

    out = os.path.join(os.path.dirname(__file__), "benchmark.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure → benchmark.png")
    plt.show()
