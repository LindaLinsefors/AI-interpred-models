"""Linear classifier probes for the "steps since" targets of the second-argmax RNN:
ONE probe per target, shared by all time steps (it is not told t).

Same network, inputs, split and targets as linear_probes.py, but each age a_k = t - p_k
is a class. Probe: multinomial logistic regression on the standardised hidden state,
    P(a = c | h) = softmax_c(W z + b),   z = (h - mu) / sigma,
fit on the training states of every step where the target is defined. Scored by accuracy
on held-out sequences, per step t and over all steps ("all t").
Reference: majority-class accuracy at each t (always guess the most common age at that t;
note this baseline knows t, the probe does not).

Run:  python steps_classifier.py [n_train] [seed]     (default 200,000 train rows, seed 0)
Writes steps_classifier.csv and age_probes.npz (weights) into this folder,
steps_classifier.png into OUT (../../archived-probed/<this folder's name>/).
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from linear_probes import HIDE, OUT, T, data, pooled

NAMES = ["steps since max", "steps since 2nd", "steps since 3rd"]


def fit(job):
    """One pooled classifier: test accuracy and majority baseline per t (+ all t), weights."""
    (Ht, yt, tt), (He, ye, te) = job
    mu, sd = Ht.mean(0), Ht.std(0)
    clf = LogisticRegression(C=10.0, max_iter=3000).fit((Ht - mu) / sd, yt)
    hit = clf.predict((He - mu) / sd) == ye
    acc, base = np.full(T + 1, np.nan), np.full(T + 1, np.nan)
    for t in np.unique(te):
        vals, counts = np.unique(yt[tt == t], return_counts=True)
        if len(vals) > 1:                                     # skip constant cells
            acc[t] = hit[te == t].mean()
            base[t] = (ye[te == t] == vals[counts.argmax()]).mean()
    vals, counts = np.unique(yt, return_counts=True)
    acc[T], base[T] = hit.mean(), (ye == vals[counts.argmax()]).mean()
    return acc, base, clf.coef_, clf.intercept_, clf.classes_, mu, sd


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    _, (Htr, Hte), (Ytr, Yte) = data(n_train, seed)
    jobs = [(pooled(Htr, Ytr[v, 3 + k]), pooled(Hte, Yte[v, 3 + k]))
            for v in range(2) for k in range(3)]
    jobs = [((H, y.astype(int), t), (He, ye.astype(int), te)) for (H, y, t), (He, ye, te) in jobs]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(fit, jobs))

    A = np.array([r[0] for r in res])                         # (6, T + 1), rows v-major
    B = np.array([r[1] for r in res])
    save = {}
    for i, (_, _, W, b, cls, mu, sd) in enumerate(res):
        save |= {f"W{i}": W, f"b{i}": b, f"classes{i}": cls, f"mu{i}": mu, f"sigma{i}": sd}
    np.savez(HIDE / "age_probes.npz", **save, n_train=n_train, seed=seed)

    rows = [f"{nm} ({'incl' if v == 0 else 'excl'} x_t)" for v in range(2) for nm in NAMES]
    fmt = lambda x: "" if np.isnan(x) else f"{x:.4f}"
    with open(HIDE / "steps_classifier.csv", "w") as f:
        cols = [f"t{t}" for t in range(T)] + ["all_t"]
        f.write("target," + ",".join(f"acc_{c}" for c in cols) + ","
                + ",".join(f"majority_{c}" for c in cols) + "\n")
        for r, a, b in zip(rows, A, B):
            f.write(r + "," + ",".join(map(fmt, a)) + "," + ",".join(map(fmt, b)) + "\n")
    print(f"{n_train} train / {n_train // 4} test sequences, seed {seed}; accuracy (majority)")
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
    ax.set_title("One linear classifier per target, shared by all t: held-out accuracy, number of\n"
                 "steps as class (majority class at that t in brackets; grey = undefined or constant)",
                 loc="left", color=INK, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "steps_classifier.png", dpi=180)


if __name__ == "__main__":
    main()
