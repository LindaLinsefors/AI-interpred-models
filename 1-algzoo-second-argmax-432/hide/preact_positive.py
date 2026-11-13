"""Fraction of inputs on which each unit's pre-activation is positive, per time step.

Pre-activation at step t: z_t = W_ih x_t + W_hh h_(t-1), and h_t = ReLU(z_t).
Inputs: independent standard-normal sequences of length 10 (the task's distribution,
as in the colleague's verify.py / analysis.py).

Run:  python preact_positive.py [n_rows] [seed]     (default 1,000,000 rows, seed 0)
Writes preact_positive.csv into hide/ and preact_positive.png into ../plots/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_weights import CMAP, GROUPS, INK, ORDER, BOUNDS, UNIT_COLOR, load

PLOTS = Path(__file__).resolve().parent.parent / "plots"  # figures go to <case study>/plots/


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    W_ih, W_hh, W_out = load()
    X = np.random.default_rng(seed).standard_normal((n, 10))

    h = np.zeros((n, 16))
    pos = np.zeros((10, 16))           # pos[t, u] = fraction with z_t[u] > 0
    for t in range(10):
        z = X[:, t:t + 1] @ W_ih.T + h @ W_hh.T
        pos[t] = (z > 0).mean(0)
        h = np.maximum(z, 0)
    pred = (h @ W_out.T).argmax(1)
    acc = (pred == np.argsort(X, 1)[:, -2]).mean()
    print(f"{n} rows, seed {seed}, task accuracy {acc:.2%}")

    pct = 100 * pos
    with open(Path(__file__).resolve().parent / "preact_positive.csv", "w") as f:
        f.write("unit,group," + ",".join(f"t{t}" for t in range(10)) + "\n")
        for u in range(16):
            g = next(name for name, units, _ in GROUPS if u in units)
            f.write(f"{u},{g}," + ",".join(f"{v:.2f}" for v in pct[:, u]) + "\n")

    print("\n% of inputs with positive pre-activation (rows = units, grouped by role)")
    print("unit  " + "".join(f"{f't={t}':>7}" for t in range(10)))
    for u in ORDER:
        print(f"{u:>4}  " + "".join(f"{v:7.1f}" for v in pct[:, u]))

    fig, ax = plt.subplots(figsize=(8.5, 8))
    M = pct[:, ORDER].T
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v - 50) > 32 else INK)
    ax.set_xticks(range(10), [str(t) for t in range(10)])
    ax.set_yticks(range(16), [str(u) for u in ORDER])
    for t in ax.get_yticklabels():
        t.set_color(UNIT_COLOR[int(t.get_text())])
        t.set_fontweight("bold")
    for b in BOUNDS:
        ax.axhline(b - 0.5, color=INK, lw=1.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("time step t")
    ax.set_ylabel("hidden unit (comparators | pulses | wave units)")
    ax.set_title(f"% of inputs where pre-activation > 0  ({n:,} standard-normal sequences)",
                 loc="left", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="% positive")
    fig.tight_layout()
    fig.savefig(PLOTS / "preact_positive.png", dpi=170)


if __name__ == "__main__":
    main()
