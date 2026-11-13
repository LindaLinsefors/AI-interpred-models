"""Linear-regression probes for the three largest values, on the 16 hidden units of the
second-argmax RNN (M_16,10). The "steps since" targets are probed with classifiers in
steps_classifier.py; targets() here defines both.

Network: h_t = ReLU(W_ih x_t + W_hh h_(t-1)), h_(-1) = 0. Inputs iid N(0,1), length 10
(the distribution used in the colleague's verify.py / analysis.py).

Targets at step t (h_t is the state just after reading x_t). Let x_(1) > x_(2) > x_(3)
be the three largest values in a window, at positions p_1, p_2, p_3:
  including x_t: window x_0..x_t      ->  max, 2nd, 3rd = x_(k);  age_k = t - p_k  (>= 0)
  excluding x_t: window x_0..x_(t-1)  ->  same definitions;       age_k = t - p_k  (>= 1)
A target is undefined (and skipped) when the window has fewer than k values.

POSITIVE-ONLY VARIANT: a state (sequence, t) is also left out of training AND scoring for
the k-th value and the k-th age whenever the k-th largest value in that target's window is
<= 0 (targets() returns NaN there). The filter is per state: a sequence whose early inputs
are negative still counts at later steps, once its k-th max is positive.

Probe (value targets only): affine least squares  y ~ w . h_t + b,  fit on a training set and scored by R^2
on a held-out test set. One probe per (target, t), plus one probe per target pooled
over all t where it is defined (column "all t").

Run:  python linear_probes.py [n_train] [seed]     (default 200,000 train rows, seed 0)
Writes probe_r2.csv into this folder and probe_r2.png into OUT
(../../<this folder's name>/).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HIDE = Path(__file__).resolve().parent         # <case study>/hide/<analysis>/: scripts, CSVs, npz
ROOT = HIDE.parent.parent                       # <case study>/ = 1-algzoo-second-argmax-432/
OUT = ROOT / HIDE.name                          # figures go to <case study>/<analysis>/
WEIGHTS = (ROOT.parent / "reverse-engineered-networks" / "1-algzoo-second-argmax-432"
           / "code" / "m16_10_weights.npz")
T = 10
NAMES = ["max", "2nd max", "3rd max"]                         # value targets probed here


def hidden_states(X, W_ih, W_hh):
    """H[t] = h_t, shape (T, n, 16)."""
    h, H = np.zeros((len(X), W_hh.shape[0])), []
    for t in range(T):
        h = np.maximum(X[:, t:t + 1] @ W_ih.T + h @ W_hh.T, 0.0)
        H.append(h)
    return np.stack(H)


def targets(X):
    """All targets, NaN where undefined or where the k-th value is <= 0."""
    return positive(raw_targets(X))


def positive(Y):
    """Drop (set to NaN) the k-th value and k-th age wherever the k-th value is <= 0."""
    Y = Y.copy()
    for k in range(3):
        drop = ~(Y[:, k] > 0)                                 # also True where already NaN
        Y[:, k][drop], Y[:, 3 + k][drop] = np.nan, np.nan
    return Y


def sel(H, y):
    """Keep the states where y is defined (H: (..., 16), y: matching leading shape)."""
    ok = ~np.isnan(y)
    return H[ok], y[ok]


def raw_targets(X):
    """Y[v, k, t] for variant v (0 = including x_t, 1 = excluding x_t), target k (NAMES
    order), step t; NaN where undefined. Shape (2, 6, T, n)."""
    n = len(X)
    Y = np.full((2, 6, T, n), np.nan)
    for v in range(2):
        for t in range(T):
            end = t + 1 - v                                   # window is x_0..x_(end-1)
            pos = np.argsort(-X[:, :end], axis=1)[:, :3]      # positions of top 3
            for k in range(pos.shape[1]):
                Y[v, k, t] = X[np.arange(n), pos[:, k]]
                Y[v, 3 + k, t] = t - pos[:, k]
    return Y


def fit_r2(Htr, ytr, Hte, yte):
    (Htr, ytr), (Hte, yte) = sel(Htr, ytr), sel(Hte, yte)
    A = np.c_[Htr, np.ones(len(Htr))]
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.c_[Hte, np.ones(len(Hte))] @ w
    return 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_test = n_train // 4
    d = np.load(WEIGHTS)
    W_ih, W_hh = d["W_ih"], d["W_hh"]

    rng = np.random.default_rng(seed)
    Xtr, Xte = rng.standard_normal((n_train, T)), rng.standard_normal((n_test, T))
    Htr, Hte = hidden_states(Xtr, W_ih, W_hh), hidden_states(Xte, W_ih, W_hh)
    Ytr, Yte = targets(Xtr), targets(Xte)

    R2 = np.full((2, 3, T + 1), np.nan)     # last column: pooled over t
    for v in range(2):
        for k in range(3):
            ts = [t for t in range(T) if (~np.isnan(Ytr[v, k, t])).any()]
            for t in ts:
                R2[v, k, t] = fit_r2(Htr[t], Ytr[v, k, t], Hte[t], Yte[v, k, t])
            R2[v, k, T] = fit_r2(Htr[ts], Ytr[v, k, ts], Hte[ts], Yte[v, k, ts])

    rows = [f"{nm} ({'incl' if v == 0 else 'excl'} x_t)" for v in range(2) for nm in NAMES]
    M = R2.reshape(6, T + 1)
    with open(HIDE / "probe_r2.csv", "w") as f:
        f.write("target," + ",".join(f"t{t}" for t in range(T)) + ",all_t\n")
        for r, m in zip(rows, M):
            f.write(r + "," + ",".join("" if np.isnan(x) else f"{x:.4f}" for x in m) + "\n")
    print(f"{n_train} train / {n_test} test rows, seed {seed}")
    for r, m in zip(rows, M):
        print(f"{r:28s}" + " ".join("  -  " if np.isnan(x) else f"{x:5.2f}" for x in m))
    plot(M)


def plot(M):
    INK, MUTED = "#222220", "#8a8a85"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ecebe8")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(T + 1), [str(t) for t in range(T)] + ["all t"])
    ax.set_yticks(range(6), NAMES * 2)
    ax.tick_params(length=0)
    ax.axhline(2.5, color="white", lw=4)
    ax.axvline(T - 0.5, color="white", lw=4)
    for y, lab in [(1, "incl. x_t"), (4, "excl. x_t")]:
        ax.text(-2.6, y, lab, rotation=90, ha="center", va="center", color=INK,
                fontsize=9, fontweight="bold")
    ax.set_xlabel("time step t (probe reads h_t)")
    ax.set_title("Linear-regression probe R² on held-out inputs, value of the k-th largest so far\n"
                 "only states where it is > 0 (grey = undefined)", loc="left", color=INK, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "probe_r2.png", dpi=180)


if __name__ == "__main__":
    main()
