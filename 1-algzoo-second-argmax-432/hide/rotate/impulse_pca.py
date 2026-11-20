"""PCA of the impulse-response trajectory (x = (1, 0, ..., 0)), pulse units removed. Directions and mean are
fitted on h_T0..h_9 only (T0 = 2: t = 0, 1 excluded); all of h_0..h_9 is plotted.
Plots PC_k vs PC_0 for k = 1..12 on a common scale. Writes rotate/impulse_pca.png."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
sys.path.insert(0, str(ROOT / "hide"))
from plot_weights import WEIGHTS, GROUPS, INK, MUTED, plt

d = np.load(WEIGHTS)
W_ih, W_hh = d["W_ih"][:, 0], d["W_hh"]
units = [u for g, us, _ in GROUPS if g != "pulses" for u in us]   # 13 non-pulse units


def run(xs):
    """Hidden states h_0..h_9 (10, 16) for input sequence xs."""
    h, H = np.zeros(16), []
    for x in xs:
        h = np.maximum(0, W_ih * x + W_hh @ h)
        H.append(h)
    return np.array(H)


T0 = 2                                         # first step used to fit the PCA (user decision, 2026-11-13)


def pca(xs=(1.0,) + (0.0,) * 9, t0=T0):
    """PCA of h_t0..h_9 for input xs (default: the impulse), non-pulse units. Returns mean, full basis Vt (13, 13), S, var.
    10 - t0 points -> at most 9 - t0 nonzero PCs; rows of Vt beyond that are an arbitrary basis of the complement."""
    X = run(xs)[t0:, units]
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=True)
    Vt *= np.where(Vt.sum(1) < 0, -1, 1)[:, None]   # sign convention: loadings sum > 0 (mostly along +neurons)
    var = np.zeros(13); var[:len(S)] = S**2 / (S**2).sum()
    return mean, Vt, S, var


def main():
    mean, Vt, S, var = pca()
    P = (run([1.0] + [0.0] * 9)[:, units] - mean) @ Vt.T                # scores (10, 13); t < T0 projected, not fitted

    lim = 1.08 * np.abs(P).max()
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.3), sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat, start=1):
        ax.plot(P[:, 0], P[:, k], color=MUTED, lw=1, zorder=1)
        ax.plot(P[T0:, 0], P[T0:, k], color=MUTED, lw=2.5, alpha=0.6, zorder=1)   # fitted steps
        sc = ax.scatter(P[:, 0], P[:, k], c=np.arange(10), cmap="viridis", s=40, zorder=2,
                        edgecolor="white", linewidth=1)
        for t in (0, 1, 9):
            ax.annotate(f"t={t}", (P[t, 0], P[t, k]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(f"PC{k} ({100 * var[k]:.1f}%) vs PC0", fontsize=9, color=INK)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, lw=0.5, zorder=0); ax.axvline(0, color=MUTED, lw=0.5, zorder=0)
    for ax in axes[-1]: ax.set_xlabel(f"PC0 ({100 * var[0]:.1f}%)")
    fig.suptitle(f"Impulse response $x=(1,0,\ldots,0)$: PCA fitted on $h_{T0}..h_9$ over the 13 non-pulse units "
                 "(mean subtracted); all $t$ projected (thick line: fitted steps)", color=INK)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="t")
    fig.savefig(ROOT / "rotate" / "impulse_pca.png", dpi=150, bbox_inches="tight")

    print("explained variance:", np.round(var, 4))
    print("singular values:", np.round(S, 3))
    for k in range(4):
        print(f"PC{k} loadings:", {u: round(v, 2) for u, v in zip(units, Vt[k])})


if __name__ == "__main__":
    main()
