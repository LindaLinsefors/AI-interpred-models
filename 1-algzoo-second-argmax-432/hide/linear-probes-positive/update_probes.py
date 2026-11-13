"""Linear classifier probes for "did the k-th max just update?", one probe per target shared
by all time steps (positive-only variant, like the other probes in this folder).

Event at step s, from the rank r of x_s among x_0..x_s (r = 1 + #{j < s : x_j > x_s}):
  lands_k    r == k   x_s is the new k-th largest           (k = 1: x_s is a new max)
  changed_k  r <= k   the k-th largest value changed        (k = 1: same as lands_1)
  shifted_k  r <  k   it changed, but x_s is not the new k-th: x_s is a new max (k = 2),
                      or a new max or new 2nd max (k = 3)

Two probes per event, both binary logistic regression on standardised features, C = 10:
  probe 1 ("just now"):    reads [h_(t-1), x_t], predicts the event at s = t  (h_(-1) = 0)
  probe 2 ("last round"):  reads h_(t-1),        predicts the event at s = t - 1
Probe 2 is scored for s = 0..9, i.e. it reads h_0..h_9.

Positive-only filter, on the window after the event: a state is kept for the k-th events
only if the k-th largest of x_0..x_s is > 0, i.e. at least k of x_0..x_s are positive.
This also drops steps with fewer than k inputs. Scored on held-out sequences per step s
and over all s, against the majority class at that s (which knows s; the probe does not).

Run:  python update_probes.py [n_train] [seed]     (default 200,000 train sequences, seed 0)
Writes update_probes.csv and update_probes.npz (weights: W1, b1, mu1, sigma1 for probe 1,
W2... for probe 2, one row per event in EVENTS order) into this folder and update_probes.png into OUT
(../../<this folder's name>/).
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from linear_probes import HIDE, OUT, T, data

EVENTS = [("lands", 1), ("lands", 2), ("changed", 2), ("shifted", 2),
          ("lands", 3), ("changed", 3), ("shifted", 3)]
LABEL = {"lands": "x_s is the new {k}", "changed": "{k} changed (rank ≤ {kk})",
         "shifted": "{k} changed, not via x_s (rank < {kk})"}
KTH = {1: "max", 2: "2nd max", 3: "3rd max"}
PROBES = ["probe 1: reads [h_(t-1), x_t], event at s = t",
          "probe 2: reads h_(t-1), event at s = t - 1"]


def events(X):
    """E[e, s, n] in {0, 1} for each event e at step s, NaN where filtered out."""
    rank = 1 + (np.triu(np.ones((T, T)), 1)[None] * (X[:, :, None] > X[:, None, :])).sum(1)
    npos = np.cumsum(X > 0, axis=1)                           # positives among x_0..x_s
    E = np.full((len(EVENTS), T, len(X)), np.nan)
    for e, (kind, k) in enumerate(EVENTS):
        hit = {"lands": rank == k, "changed": rank <= k, "shifted": rank < k}[kind]
        E[e] = np.where(npos >= k, hit, np.nan).T
    return E


def features(X, H, probe):
    """F[s, n] for event step s: probe 1 = [h_(s-1), x_s], probe 2 = h_s."""
    if probe == 1:
        Hprev = np.concatenate([np.zeros_like(H[:1]), H[:-1]])
        return np.concatenate([Hprev, X.T[:, :, None]], axis=2)
    return H


def stack(F, y):
    ok = ~np.isnan(y)
    return F[ok], y[ok].astype(int), np.nonzero(ok)[0]


def fit(job):
    (Ft, yt, st), (Fe, ye, se) = job
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-12
    clf = LogisticRegression(C=10.0, max_iter=3000).fit((Ft - mu) / sd, yt)
    hit = clf.predict((Fe - mu) / sd) == ye
    acc, base = np.full(T + 1, np.nan), np.full(T + 1, np.nan)
    for s in np.unique(se):
        if len(np.unique(yt[st == s])) > 1:                    # skip constant cells
            acc[s] = hit[se == s].mean()
            base[s] = (ye[se == s] == round(yt[st == s].mean())).mean()  # train majority
    acc[T], base[T] = hit.mean(), (ye == round(yt.mean())).mean()
    return acc, base, clf.coef_[0], clf.intercept_[0], mu, sd


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    (Xtr, Xte), (Htr, Hte), _ = data(n_train, seed)
    Etr, Ete = events(Xtr), events(Xte)
    jobs = [(stack(features(Xtr, Htr, p), Etr[e]), stack(features(Xte, Hte, p), Ete[e]))
            for p in (1, 2) for e in range(len(EVENTS))]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(fit, jobs))
    A = np.array([r[0] for r in res])                         # (2 * events, T + 1)
    B = np.array([r[1] for r in res])
    np.savez(HIDE / "update_probes.npz", events=np.array([f"{kind}_{k}" for kind, k in EVENTS]),
             **{f"{key}{p}": np.array([r[i] for r in res[(p - 1) * len(EVENTS):p * len(EVENTS)]])
                for p in (1, 2) for key, i in (("W", 2), ("b", 3), ("mu", 4), ("sigma", 5))})

    names = [LABEL[kind].format(k=KTH[k], kk=k) for kind, k in EVENTS]
    rows = [f"probe {p}: {nm}" for p in (1, 2) for nm in names]
    fmt = lambda x: "" if np.isnan(x) else f"{x:.4f}"
    with open(HIDE / "update_probes.csv", "w", encoding="utf-8") as f:
        cols = [f"s{s}" for s in range(T)] + ["all_s"]
        f.write("target," + ",".join(f"acc_{c}" for c in cols) + ","
                + ",".join(f"majority_{c}" for c in cols) + "\n")
        for r, a, b in zip(rows, A, B):
            f.write(f'"{r}",' + ",".join(map(fmt, a)) + "," + ",".join(map(fmt, b)) + "\n")
    print(f"{n_train} train / {n_train // 4} test sequences, seed {seed}; accuracy (majority)")
    for r, a, b in zip(rows, A, B):
        print(f"{r:58s}" + " ".join("     -     " if np.isnan(x)
                                    else f"{x:.2f} ({y:.2f})" for x, y in zip(a, b)))
    plot(A, B, names)


def plot(A, B, names):
    INK, MUTED = "#222220", "#8a8a85"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ecebe8")
    n = len(names)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.6))
    for p, ax in enumerate(axes):
        a, b = A[p * n:(p + 1) * n], B[p * n:(p + 1) * n]
        ax.imshow(a, cmap=cmap, vmin=0.5, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(a):
            if not np.isnan(v):
                c = "white" if v > 0.8 else INK
                ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=c)
                ax.text(j, i + 0.27, f"({b[i, j]:.2f})", ha="center", va="center",
                        fontsize=5.5, color=c, alpha=0.8)
        ax.set_xticks(range(T + 1), [str(s) for s in range(T)] + ["all s"])
        ax.set_yticks(range(n), names)
        ax.tick_params(length=0)
        for y in (0.5, 3.5):
            ax.axhline(y, color="white", lw=3)
        ax.axvline(T - 0.5, color="white", lw=4)
        ax.set_title(PROBES[p], loc="left", color=INK, fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].set_xlabel("step s of the possible update (rank of x_s among x_0..x_s)")
    fig.suptitle("Did the k-th max just update? One linear classifier per event, shared by all "
                 "steps: held-out accuracy\n(majority class at that s in brackets; colour from "
                 "0.5; grey = undefined or constant; only states where the k-th max > 0)",
                 x=0.02, ha="left", color=INK, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "update_probes.png", dpi=180)


if __name__ == "__main__":
    main()
