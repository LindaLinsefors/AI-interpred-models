"""Run x = (1,0,0,0,2,0,0,0,0,0) and project h_0..h_9 (non-pulse units) onto the impulse PCA basis
(impulse mean subtracted). Old impulse trajectory shown faintly. Writes rotate/two_pulses_on_impulse_pca.png."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from impulse_pca import T0, ROOT, INK, MUTED, plt, run, pca, units

XS_OLD = [1.0] + [0.0] * 9
XS_NEW = [1.0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0]

mean, Vt, S, var = pca()
P_old = (run(XS_OLD)[:, units] - mean) @ Vt.T          # t = 0..9; PCA fitted on the impulse at t = T0..9
P_new = (run(XS_NEW)[:, units] - mean) @ Vt.T

lim = 1.08 * np.abs(np.vstack([P_old, P_new])).max()
fig, axes = plt.subplots(3, 4, figsize=(12, 9.3), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat, start=1):
    ax.plot(P_old[:, 0], P_old[:, k], color=MUTED, lw=1, alpha=0.5, zorder=1)
    ax.plot(P_old[T0:, 0], P_old[T0:, k], color=MUTED, lw=2.5, alpha=0.6, zorder=1)   # fitted steps
    ax.scatter(P_old[:, 0], P_old[:, k], color=MUTED, s=12, alpha=0.5, zorder=1)
    ax.plot(P_new[:, 0], P_new[:, k], color=INK, lw=1, zorder=2)
    sc = ax.scatter(P_new[:, 0], P_new[:, k], c=np.arange(10), cmap="viridis", s=40, zorder=3,
                    edgecolor="white", linewidth=1)
    for t in (0, 1, 4, 9):
        ax.annotate(f"t={t}", (P_new[t, 0], P_new[t, k]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.set_title(f"PC{k} ({100 * var[k]:.1f}%) vs PC0", fontsize=9, color=INK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.5, zorder=0); ax.axvline(0, color=MUTED, lw=0.5, zorder=0)
for ax in axes[-1]: ax.set_xlabel(f"PC0 ({100 * var[0]:.1f}%)")
fig.suptitle(f"$x=(1,0,0,0,2,0,0,0,0,0)$ projected on the impulse PCA basis (fitted on $h_{T0}..h_9$, mean subtracted); "
             "$t=0$ included; grey = impulse $x=(1,0,\ldots,0)$", color=INK)
fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="t")
fig.savefig(ROOT / "rotate" / "two_pulses_on_impulse_pca.png", dpi=150, bbox_inches="tight")

np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("new scores (rows t=0..9, cols PC0..12):\n", P_new)
rank = 9 - T0                                            # nonzero PCs of the fit
r = np.linalg.norm(P_new[:, :rank], axis=1) / np.linalg.norm(P_new, axis=1)
print(f"fraction of norm in PC0..{rank - 1} (old span):", r)
print("radius in PC0-PC1 plane, old:", np.hypot(*P_old[:, :2].T), "\n new:", np.hypot(*P_new[:, :2].T))
