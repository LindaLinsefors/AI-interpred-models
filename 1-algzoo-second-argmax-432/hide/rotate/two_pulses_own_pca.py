"""PCA of h_T0..h_9 (T0 = 2 from impulse_pca; non-pulse units) for x = (1,0,0,0,2,0,0,0,0,0), projected on its own PCs
(own mean subtracted); all of h_0..h_9 plotted. Same layout as impulse_pca.png. Writes rotate/two_pulses_own_pca.png."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from impulse_pca import T0, ROOT, INK, MUTED, plt, run, pca, units

XS = [1.0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0]

mean, Vt, S, var = pca(XS)
P = (run(XS)[:, units] - mean) @ Vt.T                   # (10, 13); t < T0 projected, not fitted

lim = 1.08 * np.abs(P).max()
fig, axes = plt.subplots(3, 4, figsize=(12, 9.3), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat, start=1):
    ax.plot(P[:, 0], P[:, k], color=MUTED, lw=1, zorder=1)
    ax.plot(P[T0:, 0], P[T0:, k], color=MUTED, lw=2.5, alpha=0.6, zorder=1)   # fitted steps
    sc = ax.scatter(P[:, 0], P[:, k], c=np.arange(10), cmap="viridis", s=40, zorder=2,
                    edgecolor="white", linewidth=1)
    for t in (0, 1, 4, 9):
        ax.annotate(f"t={t}", (P[t, 0], P[t, k]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.set_title(f"PC{k} ({100 * var[k]:.1f}%) vs PC0", fontsize=9, color=INK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.5, zorder=0); ax.axvline(0, color=MUTED, lw=0.5, zorder=0)
for ax in axes[-1]: ax.set_xlabel(f"PC0 ({100 * var[0]:.1f}%)")
fig.suptitle(f"$x=(1,0,0,0,2,0,0,0,0,0)$: PCA fitted on $h_{T0}..h_9$ over the 13 non-pulse units "
             "(own basis, mean subtracted); all $t$ projected (thick line: fitted steps)", color=INK)
fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="t")
fig.savefig(ROOT / "rotate" / "two_pulses_own_pca.png", dpi=150, bbox_inches="tight")

np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("explained variance:", np.round(var, 4))
print("singular values:", np.round(S, 3))
for k in range(3):
    print(f"PC{k} loadings:", {u: round(v, 2) for u, v in zip(units, Vt[k])})
_, Vt_imp, _, _ = pca()
print("|cos| between own PCs (rows) and impulse PCs (cols), first 4:\n", np.abs(Vt[:4] @ Vt_imp[:4].T))
