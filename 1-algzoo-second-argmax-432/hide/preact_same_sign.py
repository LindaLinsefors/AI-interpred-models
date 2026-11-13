"""Co-firing grid: % of the time two units' pre-activations have the same sign.

Pre-activation at step t: z_t = W_ih x_t + W_hh h_(t-1), and h_t = ReLU(z_t).
Inputs: independent standard-normal sequences of length 10. Each cell pools all
(sequence, time step) pairs for t = 1..9, so it covers 9 * n_rows samples. Step 0 is
left out: there z_0 = W_ih x_0, so every pair's sign agreement is trivially 0% or 100%.

Run:  python preact_same_sign.py [n_rows] [seed]     (default 1,000,000 rows, seed 0)
Writes preact_same_sign.png into ../plots/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_weights import BOUNDS, CMAP, INK, ORDER, UNIT_COLOR, load

PLOTS = Path(__file__).resolve().parent.parent / "plots"  # figures go to <case study>/plots/


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    W_ih, W_hh, _ = load()
    X = np.random.default_rng(seed).standard_normal((n, 10))

    h = np.zeros((n, 16))
    SS = np.zeros((16, 16))            # sum over samples of sign(z_i) * sign(z_j)
    for t in range(10):
        z = X[:, t:t + 1] @ W_ih.T + h @ W_hh.T
        if t > 0:
            s = np.where(z > 0, 1.0, -1.0).astype(np.float32)
            SS += s.T @ s
        h = np.maximum(z, 0)
    pct = 100 * (1 + SS / (9 * n)) / 2  # fraction of samples where signs agree

    M = pct[np.ix_(ORDER, ORDER)]
    fig, ax = plt.subplots(figsize=(8.5, 7.6))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=100, aspect="equal")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v - 50) > 32 else INK)
    labels = [str(u) for u in ORDER]
    ax.set_xticks(range(16), labels)
    ax.set_yticks(range(16), labels)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(UNIT_COLOR[int(t.get_text())])
        t.set_fontweight("bold")
    for b in BOUNDS:
        ax.axhline(b - 0.5, color=INK, lw=1.2)
        ax.axvline(b - 0.5, color=INK, lw=1.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("hidden unit (comparators | pulses | wave units)")
    ax.set_ylabel("hidden unit (comparators | pulses | wave units)")
    ax.set_title(f"% of samples where two units' pre-activations have the same sign\n"
                 f"({n:,} standard-normal sequences, time steps 1-9 pooled; 50 = independent-looking)",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="% same sign")
    fig.tight_layout()
    fig.savefig(PLOTS / "preact_same_sign.png", dpi=170)


if __name__ == "__main__":
    main()
