"""PCA of the rows of W_out (10 x 16): row i is the readout of output position i, a point in unit space.

Eight versions: centred (mean row subtracted) or not | all 16 units or comparators 1, 6, 7, 8 only |
all positions 0-9 or positions 0-7 only. For each, scatter the row scores on PC_k against PC0,
for every PC with nonzero variance. Uncentred fractions are of the second moment.

Run:  python -B w_out_pca.py
Writes ../../w-out-analasys/w_out_pca.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
OUT = ROOT / HIDE.name
sys.path.insert(0, str(ROOT / "hide"))
from plot_weights import INK, MUTED, load  # noqa: E402

COMPARATORS = [1, 6, 7, 8]


def pca(X, centre):
    """Scores on every PC with nonzero variance, and each PC's variance fraction (rows = points)."""
    U, S, _ = np.linalg.svd(X - X.mean(0) if centre else X, full_matrices=False)
    keep = S > 1e-6 * S[0]          # float32 weights
    return (U * S)[:, keep], S[keep]**2 / (S**2).sum()


def main():
    W_out = load()[2]
    versions = [(c, u, p) for c in (False, True) for u in ("all", "comp") for p in (10, 8)]
    ncol = 9
    fig, axes = plt.subplots(len(versions), ncol, figsize=(2.3 * ncol, 2.5 * len(versions)))
    for row, (centre, units, npos) in zip(axes, versions):
        X = W_out[:npos, COMPARATORS if units == "comp" else slice(None)]
        Z, frac = pca(X, centre)
        name = (f"{'centred' if centre else 'uncentred'}, "
                f"{'units 1, 6, 7, 8' if units == 'comp' else 'all units'}, positions 0–{npos - 1}")
        print(f"{name:45s}", np.round(frac, 3))
        for k, ax in enumerate(row, start=1):
            if k >= Z.shape[1]:
                ax.axis("off")
                continue
            ax.plot(Z[:, 0], Z[:, k], color=MUTED, lw=0.8, zorder=1)          # positions in order
            ax.scatter(Z[:, 0], Z[:, k], color=INK, s=14, zorder=2)
            for i, (x, y) in enumerate(Z[:, [0, k]]):
                ax.annotate(str(i), (x, y), xytext=(2, 2), textcoords="offset points",
                            fontsize=7, color=INK)
            ax.axhline(0, color=MUTED, lw=0.5)
            ax.axvline(0, color=MUTED, lw=0.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(f"PC0 ({frac[0]:.1%})", fontsize=7)
            ax.set_ylabel(f"PC{k} ({frac[k]:.1%})", fontsize=7)
            ax.tick_params(labelsize=6)
        row[0].set_title(name, loc="left", color=INK, fontsize=9, x=0, ha="left")
    fig.suptitle("PCA of the rows of W_out (points = output positions, labelled; line joins them in order)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "w_out_pca.png", dpi=150)


if __name__ == "__main__":
    main()
