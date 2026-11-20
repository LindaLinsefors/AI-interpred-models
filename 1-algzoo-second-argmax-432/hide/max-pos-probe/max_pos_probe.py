"""Classifier probes for the age of the max so far, a_1 = t - p_1 (window x_0..x_t, so a_1 >= 0),
on the hidden state h_t of the second-argmax RNN. One probe shared by all t, trained only on
states where the max so far is > 0 (the default setup of linear-probes-positive, "incl x_t").

Versions m = 0..8: also leave out states with a_1 < m (classes m..9). Each version is fit
twice: on all 16 units, and without the pulse units 4, 5, 9.
Probe: multinomial logistic regression, C = 10, on z = (h - mu) / sigma; the weights are
shown raw on h_u, W[c, u] / sigma_u. Scored by held-out accuracy, per t and over all t,
against the majority class at each t (that baseline knows t, the probe does not).

The 18 fits run in parallel on Modal; each container rebuilds the data (seconds).
Run:  modal run max_pos_probe.py      (writes max_pos_probe.npz, .csv here; figure in OUT)
      python max_pos_probe.py         (redraw the figure from the .npz)
"""
import sys
from pathlib import Path

import modal
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
OUT = ROOT / HIDE.name
LP = ROOT / "hide" / "linear-probes-positive" / "linear_probes.py"
WEIGHTS = ROOT.parent / "reverse-engineered-networks" / "1-algzoo-second-argmax-432" / "code" / "m16_10_weights.npz"
T, MS, DROP = 10, range(9), (4, 5, 9)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.3.5", "scikit-learn==1.7.2", "matplotlib")
         .add_local_file(LP, "/root/linear_probes.py")
         .add_local_file(WEIGHTS, "/root/m16_10_weights.npz"))
app = modal.App("max-pos-probe", image=image)


@app.function(cpu=8, memory=16384, timeout=3600)
def fit(m, drop):
    import linear_probes as lp
    from sklearn.linear_model import LogisticRegression
    lp.WEIGHTS = "/root/m16_10_weights.npz"
    keep = [u for u in range(16) if not (drop and u in DROP)]
    _, (Htr, Hte), (Ytr, Yte) = lp.data(200_000, 0)
    (Ht, yt, tt), (He, ye, te) = [lp.pooled(H[..., keep], np.where(Y[0, 3] >= m, Y[0, 3], np.nan))
                                  for H, Y in ((Htr, Ytr), (Hte, Yte))]
    yt, ye = yt.astype(int), ye.astype(int)
    mu, sd = Ht.mean(0), Ht.std(0)
    clf = LogisticRegression(C=10.0, max_iter=3000).fit((Ht - mu) / sd, yt)
    hit = clf.predict((He - mu) / sd) == ye
    acc, base, bhit = np.full(T + 1, np.nan), np.full(T + 1, np.nan), np.zeros(len(ye), bool)
    for t in np.unique(te):
        bhit[te == t] = ye[te == t] == np.bincount(yt[tt == t]).argmax()
        acc[t], base[t] = hit[te == t].mean(), bhit[te == t].mean()
    acc[T], base[T] = hit.mean(), bhit.mean()
    W = np.full((len(clf.classes_), 16), np.nan)
    W[:, keep] = clf.coef_ / sd
    return W, clf.intercept_ - (clf.coef_ * mu / sd).sum(1), clf.classes_, acc, base


@app.local_entrypoint()
def main():
    jobs = [(m, d) for d in (False, True) for m in MS]
    res = list(fit.starmap(jobs))
    save = {}
    for i, (W, b, cls, acc, base) in enumerate(res):
        save |= {f"W{i}": W, f"b{i}": b, f"classes{i}": cls, f"acc{i}": acc, f"base{i}": base}
    np.savez(HIDE / "max_pos_probe.npz", **save, ms=np.array(list(MS)), drop=np.array(DROP))
    cols = [f"t{t}" for t in range(T)] + ["all_t"]
    with open(HIDE / "max_pos_probe.csv", "w") as f:
        f.write("units,min_age," + ",".join(f"acc_{c}" for c in cols) + ","
                + ",".join(f"majority_{c}" for c in cols) + "\n")
        for (m, d), (_, _, _, acc, base) in zip(jobs, res):
            f.write(f"{'no 4 5 9' if d else 'all'},{m}," + ",".join(f"{x:.4f}" for x in acc)
                    + "," + ",".join(f"{x:.4f}" for x in base) + "\n")
            print(f"{'no 4,5,9' if d else 'all 16  '} age>={m}: acc {acc[T]:.3f} (majority {base[T]:.3f})")
    plot()


def plot():
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "hide"))
    from plot_weights import BOUNDS, INK, ORDER, UNIT_COLOR, group_legend, heat
    P = np.load(HIDE / "max_pos_probe.npz")
    ul, uc = [str(u) for u in ORDER], [UNIT_COLOR[u] for u in ORDER]
    n = len(MS)
    fig = plt.figure(figsize=(14, 30))
    grid = fig.add_gridspec(n, 2, height_ratios=[T - m + 1.2 for m in MS])
    for d in range(2):
        for m in MS:
            i = d * n + m
            M = P[f"W{i}"][:, ORDER]
            if len(M) == 2:          # binary: sklearn's one row w is the log-odds of the later age;
                M = np.array([-M[0], M[0]]) / 2      # as the symmetric softmax pair (+-w/2)
            ax = fig.add_subplot(grid[m, d])
            heat(ax, M, ul, [f"age {c}" for c in P[f"classes{i}"]],
                 f"{'all 16 units' if d == 0 else 'without units 4, 5, 9'}, age ≥ {m}: "
                 f"acc {P[f'acc{i}'][T]:.3f} (majority {P[f'base{i}'][T]:.3f})",
                 xcolors=uc, xbounds=BOUNDS, fmt="{:.2f}", vmax=np.nanmax(np.abs(M)), fontsize=5.5)
            for txt in ax.texts:
                if txt.get_text() == "nan":
                    txt.set_text("")
            if m == n - 1:
                ax.set_xlabel("hidden unit")
    fig.suptitle("Classifier probe for steps since the max (window x_0..x_t), one probe shared by all t,\n"
                 "trained only on states where the max > 0 and age ≥ m. Raw weight W[c, u] / σ_u on h_u "
                 "(own colour scale per panel, red > 0, blue < 0)", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    group_legend(fig, y=0.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "probe_weights.png", dpi=170)
    print("wrote", OUT / "probe_weights.png")


if __name__ == "__main__":
    plot()
