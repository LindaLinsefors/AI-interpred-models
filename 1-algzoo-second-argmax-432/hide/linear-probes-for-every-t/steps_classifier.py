"""Linear classifier probes for the "steps since" targets of the second-argmax RNN.

Same network, inputs and targets as linear_probes.py, but each age a_k = t - p_k is
treated as a class rather than a number. Probe: multinomial logistic regression on the
standardised hidden state,
    P(a = c | h_t) = softmax_c(W z_t + b),   z_t = (h_t - mean) / std,
fit on a training set and scored by accuracy on a held-out test set. One probe per
(target, t), plus one per target pooled over all t where it varies (column "all t").
Reference: the majority-class accuracy on the same test cells (always guess the most
common age at that t).

Run:  python steps_classifier.py [n_train] [seed]     (default 200,000 train rows, seed 0)
Writes steps_classifier.csv into this folder and steps_classifier.png
into OUT (../../archived-probed/<this folder's name>/).
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from linear_probes import HIDE, OUT, T, WEIGHTS, hidden_states, targets

NAMES = ["steps since max", "steps since 2nd", "steps since 3rd"]


def fit_acc(job):
    Htr, ytr, Hte, yte = job
    if len(np.unique(ytr)) < 2:                                 # constant target
        return np.nan, np.nan
    sc = StandardScaler().fit(Htr)
    clf = LogisticRegression(C=10.0, max_iter=2000).fit(sc.transform(Htr), ytr)
    acc = (clf.predict(sc.transform(Hte)) == yte).mean()
    vals, counts = np.unique(ytr, return_counts=True)
    base = (yte == vals[counts.argmax()]).mean()
    return acc, base


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_test = n_train // 4
    d = np.load(WEIGHTS)
    W_ih, W_hh = d["W_ih"], d["W_hh"]

    rng = np.random.default_rng(seed)                          # same draws as linear_probes.py
    Xtr, Xte = rng.standard_normal((n_train, T)), rng.standard_normal((n_test, T))
    Htr, Hte = hidden_states(Xtr, W_ih, W_hh), hidden_states(Xte, W_ih, W_hh)
    Ytr, Yte = targets(Xtr)[:, 3:], targets(Xte)[:, 3:]        # ages only: (2, 3, T, n)

    cells, jobs = [], []
    for v in range(2):
        for k in range(3):
            ts = [t for t in range(T) if not np.isnan(Ytr[v, k, t, 0])]
            for t in ts:
                cells.append((v, k, t))
                jobs.append((Htr[t], Ytr[v, k, t].astype(int), Hte[t], Yte[v, k, t].astype(int)))
            cells.append((v, k, T))
            jobs.append((np.concatenate(Htr[ts]), Ytr[v, k, ts].ravel().astype(int),
                         np.concatenate(Hte[ts]), Yte[v, k, ts].ravel().astype(int)))
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(fit_acc, jobs))

    ACC, BASE = np.full((2, 3, T + 1), np.nan), np.full((2, 3, T + 1), np.nan)
    for (v, k, t), (a, b) in zip(cells, res):
        ACC[v, k, t], BASE[v, k, t] = a, b

    rows = [f"{nm} ({'incl' if v == 0 else 'excl'} x_t)" for v in range(2) for nm in NAMES]
    A, B = ACC.reshape(6, T + 1), BASE.reshape(6, T + 1)
    fmt = lambda x: "" if np.isnan(x) else f"{x:.4f}"
    with open(HIDE / "steps_classifier.csv", "w") as f:
        cols = [f"t{t}" for t in range(T)] + ["all_t"]
        f.write("target," + ",".join(f"acc_{c}" for c in cols) + ","
                + ",".join(f"majority_{c}" for c in cols) + "\n")
        for r, a, b in zip(rows, A, B):
            f.write(r + "," + ",".join(map(fmt, a)) + "," + ",".join(map(fmt, b)) + "\n")
    print(f"{n_train} train / {n_test} test rows, seed {seed}; accuracy (majority baseline)")
    for r, a, b in zip(rows, A, B):
        print(f"{r:28s}" + " ".join("     -     " if np.isnan(x)
                                    else f"{x:.2f} ({y:.2f})" for x, y in zip(a, b)))
    plot(A, B)


def plot(A, B):
    INK, MUTED = "#222220", "#8a8a85"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ecebe8")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.imshow(A, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(A):
        if not np.isnan(v):
            c = "white" if v > 0.6 else INK
            ax.text(j, i - 0.1, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=c)
            ax.text(j, i + 0.25, f"({B[i, j]:.2f})", ha="center", va="center", fontsize=5.5,
                    color=c, alpha=0.8)
    ax.set_xticks(range(T + 1), [str(t) for t in range(T)] + ["all t"])
    ax.set_yticks(range(6), NAMES * 2)
    ax.tick_params(length=0)
    ax.axhline(2.5, color="white", lw=4)
    ax.axvline(T - 0.5, color="white", lw=4)
    for y, lab in [(1, "incl. x_t"), (4, "excl. x_t")]:
        ax.text(-3.4, y, lab, rotation=90, ha="center", va="center", color=INK,
                fontsize=9, fontweight="bold")
    ax.set_xlabel("time step t (probe reads h_t)")
    ax.set_title("Linear classifier probe accuracy on held-out inputs, number of steps as class\n"
                 "(majority-class baseline in brackets; grey = undefined or constant)",
                 loc="left", color=INK, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "steps_classifier.png", dpi=180)


if __name__ == "__main__":
    main()
