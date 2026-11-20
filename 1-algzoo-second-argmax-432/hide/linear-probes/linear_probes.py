"""Linear-regression probes for the three largest values, on the 16 hidden units of the
second-argmax RNN (M_16,10). One probe per target, shared by all time steps. The
"steps since" targets are probed with classifiers in steps_classifier.py; targets() here
defines both, and data() builds the shared train/test sets.

Network: h_t = ReLU(W_ih x_t + W_hh h_(t-1)), h_(-1) = 0. Inputs iid N(0,1), length 10
(the distribution used in the colleague's verify.py / analysis.py).

Targets at step t (h_t is the state just after reading x_t). Let x_(1) > x_(2) > x_(3)
be the three largest values in a window, at positions p_1, p_2, p_3:
  including x_t: window x_0..x_t      ->  max, 2nd, 3rd = x_(k);  age_k = t - p_k  (>= 0)
  excluding x_t: window x_0..x_(t-1)  ->  same definitions;       age_k = t - p_k  (>= 1)
A target is undefined (and skipped) when the window has fewer than k values.

Probe: affine least squares  y ~ w . h + b,  ONE (w, b) per target, fit on the training
states of every step t where the target is defined (the probe is not told t). Scored by
R^2 on held-out sequences: per step t (on that step's rows, against that step's variance)
and over all steps together ("all t").

Run:  python linear_probes.py [n_train] [seed]     (default 200,000 train rows, seed 0)
Writes probe_r2.csv and value_probes.npz (weights) into this folder, probe_r2.png into OUT
(../../<this folder's name>/).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HIDE = Path(__file__).resolve().parent         # <case study>/hide/<analysis>/: scripts, CSVs, npz
ROOT = HIDE.parent.parent                       # <case study>/ = 1-algzoo-second-argmax-432/
OUT = ROOT / "archived-probed" / HIDE.name     # figures go to <case study>/archived-probed/<analysis>/
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
    """Y[v, k, t] for variant v (0 = including x_t, 1 = excluding x_t), target k (0-2 the
    values, 3-5 the ages, in rank order), step t; NaN where undefined. Shape (2, 6, T, n)."""
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


def data(n_train, seed):
    """Train and test sets (test = n_train // 4 sequences): inputs, states, targets."""
    W = np.load(WEIGHTS)
    rng = np.random.default_rng(seed)
    Xtr, Xte = rng.standard_normal((n_train, T)), rng.standard_normal((n_train // 4, T))
    H = [hidden_states(X, W["W_ih"], W["W_hh"]) for X in (Xtr, Xte)]
    return (Xtr, Xte), H, [targets(X) for X in (Xtr, Xte)]


def pooled(H, y):
    """Stack the steps where y (shape (T, n)) is defined: features, targets, step index."""
    ts = [t for t in range(T) if not np.isnan(y[t, 0])]
    n = H.shape[1]
    return np.concatenate(H[ts]), y[ts].ravel(), np.repeat(ts, n)


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    _, (Htr, Hte), (Ytr, Yte) = data(n_train, seed)

    R2 = np.full((2, 3, T + 1), np.nan)                       # last column: all t together
    Wts, B, SIG = np.zeros((2, 3, 16)), np.zeros((2, 3)), np.zeros((2, 3, 16))
    for v in range(2):
        for k in range(3):
            Ht, yt, _ = pooled(Htr, Ytr[v, k])
            He, ye, te = pooled(Hte, Yte[v, k])
            w, *_ = np.linalg.lstsq(np.c_[Ht, np.ones(len(Ht))], yt, rcond=None)
            Wts[v, k], B[v, k], SIG[v, k] = w[:-1], w[-1], Ht.std(0)
            pred = He @ w[:-1] + w[-1]
            for t in np.unique(te):
                R2[v, k, t] = r2(ye[te == t], pred[te == t])
            R2[v, k, T] = r2(ye, pred)
    np.savez(HIDE / "value_probes.npz", w=Wts, b=B, sigma=SIG, n_train=n_train, seed=seed)

    rows = [f"{nm} ({'incl' if v == 0 else 'excl'} x_t)" for v in range(2) for nm in NAMES]
    M = R2.reshape(6, T + 1)
    with open(HIDE / "probe_r2.csv", "w") as f:
        f.write("target," + ",".join(f"t{t}" for t in range(T)) + ",all_t\n")
        for r, m in zip(rows, M):
            f.write(r + "," + ",".join("" if np.isnan(x) else f"{x:.4f}" for x in m) + "\n")
    print(f"{n_train} train / {n_train // 4} test sequences, seed {seed}")
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
    ax.set_title("One linear-regression probe per target, shared by all t: held-out R², value of\n"
                 "the k-th largest so far (grey = undefined; colour clipped at 0)",
                 loc="left", color=INK, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "probe_r2.png", dpi=180)


if __name__ == "__main__":
    main()
