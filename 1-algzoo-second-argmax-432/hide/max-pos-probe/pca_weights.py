"""PCA of the weight matrix of the top-left panel of probe_weights.png (all 16 units, age >= 0),
with the rows for ages 0 and 1 removed: M[c, u] = W[c, u] / sigma_u for ages c = 2..9.
  rows:    8 points (ages) in R^16, centred over ages
  columns: 16 points (units) in R^8, centred over units
Plots PC1 vs PCk, k = 2..7, for each (the 8 centred rows have only 7 PCs).

Run:  python pca_weights.py      (needs max_pos_probe.npz)
Writes pca_weights.png into OUT.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from max_pos_probe import HIDE, OUT, ROOT

sys.path.insert(0, str(ROOT / "hide"))
from plot_weights import INK, MUTED, UNIT_COLOR, group_legend  # noqa: E402


def pca(X):
    """Scores on every PC and the variance fraction of every PC (rows = points)."""
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return U * S, S**2 / (S**2).sum()


def main():
    P = np.load(HIDE / "max_pos_probe.npz")
    keep = P["classes0"] >= 2
    M, ages = P["W0"][keep], P["classes0"][keep]
    K = range(1, 7)                                           # second axis: PC2..PC7
    fig, axes = plt.subplots(2, len(K), figsize=(4 * len(K), 8.6))
    for row, X, labels, colors, what in [
            (axes[0], M, [str(c) for c in ages], [INK] * len(ages), "rows: ages 2–9 (labels)"),
            (axes[1], M.T, [str(u) for u in range(16)], [UNIT_COLOR[u] for u in range(16)],
             "columns: units (labels)")]:
        Z, frac = pca(X)
        for ax, k in zip(row, K):
            if X is M:
                ax.plot(Z[:, 0], Z[:, k], color=MUTED, lw=1, zorder=1)    # ages in order
            ax.scatter(Z[:, 0], Z[:, k], c=colors, s=30, zorder=2)
            for x, y, s, c in zip(Z[:, 0], Z[:, k], labels, colors):
                ax.annotate(s, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8, color=c)
            ax.axhline(0, color=MUTED, lw=0.5)
            ax.axvline(0, color=MUTED, lw=0.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(f"PC1 ({frac[0]:.1%})")
            ax.set_ylabel(f"PC{k + 1} ({frac[k]:.1%})")
        row[0].set_title(f"PCA of the {what}", loc="left", color=INK, fontsize=10)
        print(what, "variance fractions:", np.round(frac, 3))
    fig.suptitle("Steps-since-max probe (all 16 units, age ≥ 0), raw weights W[c, u] / σ_u, "
                 "rows for ages 0 and 1 removed", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    group_legend(fig, y=0.0)
    fig.savefig(OUT / "pca_weights.png", dpi=170)


if __name__ == "__main__":
    main()
