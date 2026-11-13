"""The shared probes of linear_probes.py and steps_classifier.py (positive-only variant),
scored per "steps since the k-th largest value last changed" instead of per t.

The k-th largest value of the window changes whenever a new input lands at rank <= k, so
its last change is at the latest of the top-k positions:
    u_k = t - max(p_1, ..., p_k) = min(a_1, ..., a_k),
with a_j = t - p_j the age of the j-th largest (including x_t: window x_0..x_t, u_k >= 0;
excluding x_t: window x_0..x_(t-1), u_k >= 1). The probe for the k-th value or the k-th
age is scored on held-out states grouped by its own u_k. The probes are NOT refit: they
are the same weights (value_probes.npz, age_probes.npz), and still not told t or u_k.
R^2 per bin uses that bin's own variance. The classifier baseline is the majority class
in that bin (from training states). Because u_k = min(a_1..a_k), u_1 = a_1 fixes the age
of the max exactly (baseline 1.00), and u_2, u_3 partly fix the other ages.
Bins with fewer than MIN_N held-out states are left grey.

Run:  python by_update_age.py      (run linear_probes.py and steps_classifier.py first)
Writes probe_r2_by_update.png and steps_classifier_by_update.png into OUT
(../../<this folder's name>/), and by_update_age.csv into this folder.
"""
import matplotlib.pyplot as plt
import numpy as np

from linear_probes import HIDE, NAMES, OUT, T, data, r2

AGE_NAMES = ["steps since max", "steps since 2nd", "steps since 3rd"]
MIN_N = 100
INK, MUTED = "#222220", "#8a8a85"


def update_age(Y, v, k):
    """u_k = min(a_1..a_k) for variant v, shape (T, n); NaN where the k-th target is dropped."""
    u = np.min(Y[v, 3:3 + k + 1], axis=0)                     # ages a_1..a_(k+1), k 0-based
    return np.where(np.isnan(Y[v, k]), np.nan, u)


def by_bin(u, score):
    """Apply score(mask) to each bin u = 0..T-1 with at least MIN_N states, plus all states."""
    out = np.full(T + 1, np.nan)
    for b in range(T):
        m = u == b
        if m.sum() >= MIN_N:
            out[b] = score(m)
    out[T] = score(np.ones_like(u, bool))
    return out


def main():
    V, A = np.load(HIDE / "value_probes.npz"), np.load(HIDE / "age_probes.npz")
    _, (Htr, Hte), (Ytr, Yte) = data(int(V["n_train"]), int(V["seed"]))
    R2, ACC, BASE = (np.full((6, T + 1), np.nan) for _ in range(3))
    for v in range(2):
        for k in range(3):
            i = 3 * v + k
            ok = ~np.isnan(Yte[v, k])
            H, u = Hte[ok], update_age(Yte, v, k)[ok]
            # value probe
            y = Yte[v, k][ok]
            pred = H @ V["w"][v, k] + V["b"][v, k]
            R2[i] = by_bin(u, lambda m: r2(y[m], pred[m]))
            # age classifier, same kept states (the k-th value and age share the filter)
            a = Yte[v, 3 + k][ok].astype(int)
            z = (H - A[f"mu{i}"]) / A[f"sigma{i}"]
            guess = A[f"classes{i}"][np.argmax(z @ A[f"W{i}"].T + A[f"b{i}"], axis=1)]
            ACC[i] = by_bin(u, lambda m: (guess[m] == a[m]).mean())
            # majority class per bin, from the training states
            okt = ~np.isnan(Ytr[v, k])
            ut, at = update_age(Ytr, v, k)[okt], Ytr[v, 3 + k][okt].astype(int)
            for b in range(T):
                m = u == b
                if m.sum() >= MIN_N:
                    BASE[i, b] = (a[m] == np.bincount(at[ut == b]).argmax()).mean()
            BASE[i, T] = (a == np.bincount(at).argmax()).mean()

    rows = [f"{nm} ({'incl' if v == 0 else 'excl'} x_t)" for v in range(2) for nm in NAMES]
    fmt = lambda x: "" if np.isnan(x) else f"{x:.4f}"
    with open(HIDE / "by_update_age.csv", "w", encoding="utf-8") as f:
        cols = [f"u{b}" for b in range(T)] + ["all"]
        f.write("k-th value,metric," + ",".join(cols) + "\n")
        for name, M in (("R2", R2), ("acc", ACC), ("majority", BASE)):
            for r, m in zip(rows, M):
                f.write(f"{r},{name}," + ",".join(map(fmt, m)) + "\n")
    heat(R2, None, NAMES, "probe_r2_by_update.png",
         "Shared linear-regression probes (same as probe_r2.png): held-out R² of the k-th largest "
         "value,\nper steps since that value last changed (only states where it is > 0;\n"
         f"grey = undefined or < {MIN_N} states; colour clipped at 0)")
    heat(ACC, BASE, AGE_NAMES, "steps_classifier_by_update.png",
         "Shared linear classifiers (same as steps_classifier.png): held-out accuracy for the "
         "steps since the k-th max,\nper steps since the k-th value last changed (majority class "
         "in that bin in brackets: u_1 = steps since max,\nso that row's baseline is 1.00; only "
         f"states where the k-th max > 0; grey = undefined or < {MIN_N} states)")


def heat(M, B, names, fname, title):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ecebe8")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if not np.isnan(v):
            c = "white" if v > 0.6 else INK
            if B is None:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=c)
            else:
                ax.text(j, i - 0.1, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=c)
                ax.text(j, i + 0.25, f"({B[i, j]:.2f})", ha="center", va="center",
                        fontsize=5.5, color=c, alpha=0.8)
    ax.set_xticks(range(T + 1), [str(b) for b in range(T)] + ["all"])
    ax.set_yticks(range(6), names * 2)
    ax.tick_params(length=0)
    ax.axhline(2.5, color="white", lw=4)
    ax.axvline(T - 0.5, color="white", lw=4)
    for y, lab in [(1, "incl. x_t"), (4, "excl. x_t")]:
        ax.text(-3.4, y, lab, rotation=90, ha="center", va="center", color=INK,
                fontsize=9, fontweight="bold")
    ax.set_xlabel("u_k = steps since the k-th largest value last changed = min(a_1, ..., a_k)")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=180)
    print("wrote", fname)


if __name__ == "__main__":
    main()
