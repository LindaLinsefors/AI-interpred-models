"""Linear-regression probe for the positive record increment, on the 16 hidden units of the
second-argmax RNN (M_16,10). One probe, shared by all time steps.

Target at step t (h_t is the state just after reading x_t):
  d_t = ReLU(x_t - max(0, x_0, ..., x_(t-1))),   so d_0 = ReLU(x_0).

UNLIKE THE OTHER PROBES IN THIS FOLDER, NO STATE IS LEFT OUT: every sequence counts at every
step t = 0..9, in both training and scoring.

Probe: affine least squares  d ~ w . h + b, fit on the pooled training states (the probe is
not told t). Two controls use the same fit on fewer units: unit 4 alone, and all units but
unit 4. Scored on held-out sequences per step t and over all steps, by R^2 on all states and
on the states with d_t > 0, and by the rms prediction on the states with d_t = 0.

Run:  python increment_probe.py [n_train] [seed]     (default 200,000 train rows, seed 0)
Writes increment_probe.csv and increment_probe.npz (weights) into this folder and
increment_probe.png into OUT (../../<this folder's name>/).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from linear_probes import HIDE, OUT, ROOT, T, data, r2

sys.path.insert(0, str(ROOT / "hide"))                 # for plot_weights.py
from plot_weights import BOUNDS, INK, ORDER, UNIT_COLOR, group_legend, heat  # noqa: E402

MUTED, BLUE, ORANGE = "#8a8a85", "#2a6fbb", "#c8641e"
TIME = LinearSegmentedColormap.from_list("time", ["#a9cdf2", "#3b78c4", "#0b2a5c"])  # one hue
UNITS = {"all 16 units": np.arange(16), "unit 4 only": np.array([4]),
         "all but unit 4": np.delete(np.arange(16), 4)}


def increment(X):
    """d[t, n] = ReLU(x_t - max(0, x_0..x_(t-1)))."""
    record = np.maximum.accumulate(np.c_[np.zeros(len(X)), X[:, :-1]], axis=1)
    return np.maximum(X - record, 0).T


def fit(F, y):
    w, *_ = np.linalg.lstsq(np.c_[F, np.ones(len(F))], y, rcond=None)
    return w[:-1], w[-1]


def scores(y, pred, t):
    """Rows (R^2 on all states, R^2 on d > 0, rms prediction on d = 0), columns t = 0..9, all t."""
    masks = [t == s for s in range(T)] + [np.ones(len(y), bool)]
    return np.array([[r2(y[m], pred[m]) for m in masks],
                     [r2(y[m & (y > 0)], pred[m & (y > 0)]) for m in masks],
                     [np.sqrt((pred[m & (y == 0)] ** 2).mean()) if (m & (y == 0)).any() else np.nan
                      for m in masks]])


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    (Xtr, Xte), (Htr, Hte), _ = data(n_train, seed)
    Ftr, ytr = Htr.reshape(-1, 16), increment(Xtr).ravel()
    Fte, yte = Hte.reshape(-1, 16), increment(Xte).ravel()
    t = np.repeat(np.arange(T), len(Xte))

    S = {}
    for name, u in UNITS.items():
        w, b = fit(Ftr[:, u], ytr)
        pred = Fte[:, u] @ w + b
        S[name] = scores(yte, pred, t)
        if name == "all 16 units":
            W, B, P = w, b, pred
        print(f"{name:15s} w = {np.round(w, 4) if len(u) == 1 else '(see npz)'}  b = {b:.4f}")
    np.savez(HIDE / "increment_probe.npz", w=W, b=B, n_train=n_train, seed=seed)

    KIND = ["R2 all states", "R2 d>0", "rms pred d=0"]
    with open(HIDE / "increment_probe.csv", "w") as f:
        f.write("probe,score," + ",".join(f"t{s}" for s in range(T)) + ",all_t\n")
        for name, M in S.items():
            for kind, row in zip(KIND, M):
                f.write(f"{name},{kind}," + ",".join("" if np.isnan(x) else f"{x:.4f}" for x in row) + "\n")
                print(f"{name:15s} {kind:14s}" + " ".join(f"{x:6.3f}" for x in row))
    print("weights:", dict(zip(range(16), np.round(W, 4))), "b =", round(float(B), 4))
    print(f"share of held-out states with d > 0: {(yte > 0).mean():.3f}")
    plot(yte, P, t, S["all 16 units"], W)


def plot(y, pred, t, M, W):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(10.5, 7.3), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))     # room for the group legend
    grid = fig.add_gridspec(2, 2, height_ratios=[5, 1.5])

    ax = fig.add_subplot(grid[0, 0])
    i = np.random.default_rng(0).permutation(len(y))           # draw in random order, not by t
    ax.scatter(y[i], pred[i], c=t[i], cmap=TIME, vmin=0, vmax=T - 1, s=0.6, lw=0, alpha=0.2,
               rasterized=True)
    ax.plot([-0.3, 4.2], [-0.3, 4.2], color=MUTED, lw=1, ls="--")
    ax.text(0.04, 0.95, f"R² = {M[0, T]:.3f} (all states)\nR² = {M[1, T]:.3f} (d_t > 0)",
            transform=ax.transAxes, va="top", color=INK)
    ax.set(xlim=(-0.3, 4.2), ylim=(-0.3, 4.2), xlabel="true d_t", ylabel="probe prediction",
           aspect="equal")
    ax.set_title("Held-out states, all t (dashed = y = x)", loc="left", color=INK)
    ax.grid(color="#ecebe8", lw=0.6)
    ax.set_axisbelow(True)
    cb = fig.colorbar(ScalarMappable(Normalize(-0.5, T - 0.5), TIME.resampled(T)), ax=ax,
                      shrink=0.75, ticks=range(T))
    cb.set_label("time step t", color=INK)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    for row, color, marker, label in [(0, BLUE, "o", "all states"), (1, ORANGE, "s", "states with d_t > 0")]:
        ax.plot(range(T), M[row, :T], color=color, marker=marker, ms=5, lw=2, label=label)
    ax.set(xticks=range(T), xlabel="time step t (probe reads h_t)", ylabel="held-out R²")
    ax.set_title("R² of the shared probe at each step", loc="left", color=INK)
    ax.grid(axis="y", color="#ecebe8", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    heat(fig.add_subplot(grid[1, :]), W[ORDER][None], [str(u) for u in ORDER], ["d_t"],
         "Probe weights: raw w_u (change in the prediction per unit of h_u)",
         xcolors=[UNIT_COLOR[u] for u in ORDER], xbounds=BOUNDS, fmt="{:.3f}", fontsize=7)
    fig.suptitle("One linear probe for d_t = ReLU(x_t − max(0, x_0, …, x_(t−1))), shared by all t; "
                 "no states left out", x=0.02, ha="left", color=INK, fontsize=11)
    group_legend(fig, y=0.0)
    fig.savefig(OUT / "increment_probe.png", dpi=180)
    print("wrote increment_probe.png")


if __name__ == "__main__":
    main()
