"""Scatter of true value vs linear-probe prediction for the max, 2nd and 3rd max, on the
held-out (validation) sequences only. One probe per target, shared by all t (fit by
linear_probes.py, read from value_probes.npz); points from every step t where the
target is defined, coloured by t.

Run:  python scatter_values.py      (run linear_probes.py first; same n_train and seed)
Writes scatter_values.png into OUT (../../<this folder's name>/).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from linear_probes import HIDE, NAMES, OUT, T, data, pooled

INK, MUTED = "#222220", "#8a8a85"
TIME = LinearSegmentedColormap.from_list("time", ["#a9cdf2", "#3b78c4", "#0b2a5c"])  # one hue


def main():
    P = np.load(HIDE / "value_probes.npz")
    _, (_, Hte), (_, Yte) = data(int(P["n_train"]), int(P["seed"]))
    order = np.random.default_rng(0).permutation  # draw points in random order, not by t

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4), sharex=True, sharey=True,
                             layout="constrained")
    for v, row in enumerate(axes):
        for k, ax in enumerate(row):
            H, y, t = pooled(Hte, Yte[v, k])
            pred = H @ P["w"][v, k] + P["b"][v, k]
            r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            i = order(len(y))
            ax.scatter(y[i], pred[i], c=t[i], cmap=TIME, vmin=0, vmax=T - 1, s=0.6, lw=0,
                       alpha=0.15, rasterized=True)
            ax.plot([-4, 4.5], [-4, 4.5], color=MUTED, lw=1, ls="--")
            ax.text(0.04, 0.95, f"R² = {r2:.3f} (all t)", transform=ax.transAxes, va="top",
                    color=INK)
            ax.set_title(f"{NAMES[k]} ({'incl.' if v == 0 else 'excl.'} x_t)", color=INK)
            ax.set_xlim(-3.2, 4.2)
            ax.set_ylim(-3.2, 4.2)
            ax.set_aspect("equal")
            ax.grid(color="#ecebe8", lw=0.6)
            ax.set_axisbelow(True)
            if v == 1:
                ax.set_xlabel("true value")
            if k == 0:
                ax.set_ylabel("probe prediction")
    sm = ScalarMappable(Normalize(-0.5, T - 0.5), TIME.resampled(T))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, ticks=range(T))
    cb.set_label("time step t", color=INK)
    cb.outline.set_visible(False)
    fig.suptitle("One linear-regression probe per target, shared by all t: held-out sequences only "
                 "(dashed = y = x)", x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "scatter_values.png", dpi=180, bbox_inches="tight")
    print("wrote scatter_values.png")


if __name__ == "__main__":
    main()
