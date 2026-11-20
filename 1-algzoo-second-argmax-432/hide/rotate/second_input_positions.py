"""x = (1, 0, ..., 2 at position p, ..., 0) for p = 1..9, projected on the impulse PCA basis
(fitted on the impulse's h_T0..h_9, T0 = 2 from impulse_pca, non-pulse units, impulse mean subtracted). PC1 vs PC0 only,
t = 0..9, one panel per p; grey = impulse; dotted = x = (2, 0, ..., 0); the new run is drawn only from t = p - 1 (last shared state) on.
Writes rotate/second_input_positions_pc01.png."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from impulse_pca import T0, ROOT, INK, MUTED, plt, run, pca, units

mean, Vt, S, var = pca(t0=T0)
proj = lambda xs: (run(xs)[:, units] - mean) @ Vt[:2].T          # (10, 2): PC0, PC1
P_old = proj([1.0] + [0.0] * 9)
P_two = proj([2.0] + [0.0] * 9)                                  # doubled impulse, for reference
P_new = {p: proj([1.0] + [0.0] * (p - 1) + [2.0] + [0.0] * (9 - p)) for p in range(1, 10)}

lim = 1.08 * np.abs(np.vstack([P_old, P_two, *P_new.values()])).max()
fig, axes = plt.subplots(3, 3, figsize=(10, 10.4), sharex=True, sharey=True)
for ax, (p, P) in zip(axes.flat, P_new.items()):
    ax.plot(*P_old.T, color=MUTED, lw=1, alpha=0.5, zorder=1)
    ax.plot(*P_old[T0:].T, color=MUTED, lw=2.5, alpha=0.6, zorder=1)   # steps the PCA was fitted on
    ax.scatter(*P_old.T, color=MUTED, s=12, alpha=0.5, zorder=1)
    ax.plot(*P_two.T, color=MUTED, lw=1.2, ls=":", zorder=1)
    # new run only from where it diverges: identical to the impulse for t < p, so draw from the shared point t = p - 1
    ax.plot(*P[p - 1:].T, color=INK, lw=1, zorder=2)
    sc = ax.scatter(*P[p:].T, c=np.arange(p, 10), cmap="viridis", vmin=0, vmax=9, s=40, zorder=3,
                    edgecolor="white", linewidth=1)
    for t in sorted({p, 9}):
        ax.annotate(f"t={t}", P[t], xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
    xs = ["1"] + ["0"] * 9; xs[p] = "2"
    ax.set_title(f"x = ({','.join(xs)})", fontsize=9, color=INK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.5, zorder=0); ax.axvline(0, color=MUTED, lw=0.5, zorder=0)
for ax in axes[-1]: ax.set_xlabel(f"PC0 ({100 * var[0]:.1f}%)")
for ax in axes[:, 0]: ax.set_ylabel(f"PC1 ({100 * var[1]:.1f}%)")
fig.suptitle(f"Second input 2 at position p, projected on the impulse PCA basis (fitted on $h_{T0}..h_9$, mean subtracted)\n"
             r"grey = impulse $x=(1,0,\ldots,0)$ (thick: fitted steps); dotted = $x=(2,0,\ldots,0)$; new run shown from its divergence at $t=p$", color=INK)
fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label="t")
fig.savefig(ROOT / "rotate" / "second_input_positions_pc01.png", dpi=150, bbox_inches="tight")
print("explained variance:", var[:4].round(4))
print("h0 point:", P_old[0].round(1))
for p, P in P_new.items(): print(p, "at t=p:", P[p].round(1))
