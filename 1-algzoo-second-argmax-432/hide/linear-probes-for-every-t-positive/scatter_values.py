"""Scatter of true value vs linear-probe prediction for the max, 2nd and 3rd max,
on the held-out (validation) set only, at one time step t (default 9, the last step).

Same network, data, split and probes as linear_probes.py: affine least squares
y ~ w . h_t + b fit on the training rows; only the held-out rows are plotted.

Run:  python scatter_values.py [t] [n_train] [seed]     (default t = 9, 200,000, seed 0)
Writes scatter_values_t{t}.png into OUT (../../<this folder's name>/).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from linear_probes import NAMES, OUT, T, WEIGHTS, hidden_states, sel, targets

INK, MUTED, DOT = "#222220", "#8a8a85", "#1f4e9c"


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    d = np.load(WEIGHTS)
    rng = np.random.default_rng(seed)                          # same draws as linear_probes.py
    Xtr, Xte = rng.standard_normal((n_train, T)), rng.standard_normal((n_train // 4, T))
    Htr = hidden_states(Xtr, d["W_ih"], d["W_hh"])[t]
    Hte = hidden_states(Xte, d["W_ih"], d["W_hh"])[t]
    Ytr, Yte = targets(Xtr), targets(Xte)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.6), sharex=True, sharey=True)
    for v, row in enumerate(axes):
        for k, ax in enumerate(row):
            (Ht, ytr), (He, yte) = sel(Htr, Ytr[v, k, t]), sel(Hte, Yte[v, k, t])
            if len(ytr) == 0:
                ax.set_axis_off()
                continue
            w, *_ = np.linalg.lstsq(np.c_[Ht, np.ones(len(Ht))], ytr, rcond=None)
            pred = np.c_[He, np.ones(len(He))] @ w
            r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
            ax.scatter(yte, pred, s=1, lw=0, alpha=0.08, color=DOT, rasterized=True)
            ax.plot([-2, 4.2], [-2, 4.2], color=MUTED, lw=1, ls="--")
            ax.text(0.04, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top", color=INK)
            ax.set_title(f"{NAMES[k]} of x_0..x_{t if v == 0 else t - 1}", color=INK)
            ax.set_xlim(-0.6, 4.1)
            ax.set_ylim(-0.6, 4.1)
            ax.set_aspect("equal")
            ax.grid(color="#ecebe8", lw=0.6)
            ax.set_axisbelow(True)
            if v == 1:
                ax.set_xlabel("true value")
            if k == 0:
                ax.set_ylabel("probe prediction")
    fig.suptitle(f"Linear-regression probes on h_{t}: held-out inputs with k-th max > 0 only "
                 f"({len(Xte):,} sequences; dashed = y = x)", x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"scatter_values_t{t}.png", dpi=180)
    print(f"wrote scatter_values_t{t}.png")


if __name__ == "__main__":
    main()
