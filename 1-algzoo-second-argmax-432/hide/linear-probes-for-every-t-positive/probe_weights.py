"""Weights of the linear probes at the last step (default t = 9), per hidden unit.

Same network, data, split and probes as linear_probes.py / steps_classifier.py.

  values (max, 2nd, 3rd):  y ~ w . h_t + b.  Shown as w_u * sigma_u (sigma_u = std of unit u
                           over the training rows): the change in the prediction per one
                           standard deviation of unit u. Units that are always 0 get weight 0.
  steps since (a_1..a_3):  softmax_c(W z_t + b), z_t = (h_t - mu) / sigma. Shown as W[c, u],
                           one row per class c (age). Softmax only sees differences between
                           rows; the L2 penalty picks the solution whose rows sum to ~0.

Units are ordered and coloured by group as in ../plot_weights.py (comparators,
pulses, wave units). Each panel has its own colour scale (red > 0, blue < 0).

Run:  python probe_weights.py [t] [n_train] [seed]     (default t = 9, 200,000, seed 0)
Writes probe_weights_t{t}.png into OUT (../../<this folder's name>/).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from linear_probes import NAMES, OUT, ROOT, T, WEIGHTS, hidden_states, sel, targets

sys.path.insert(0, str(ROOT / "hide"))                 # for plot_weights.py
from plot_weights import BOUNDS, INK, ORDER, UNIT_COLOR, group_legend, heat  # noqa: E402

AGE_NAMES = ["steps since max", "steps since 2nd", "steps since 3rd"]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    d = np.load(WEIGHTS)
    rng = np.random.default_rng(seed)                          # same draws as linear_probes.py
    X = rng.standard_normal((n_train, T))
    H = hidden_states(X, d["W_ih"], d["W_hh"])[t][:, ORDER]    # columns in group order
    Y = targets(X)

    ul, uc = [str(u) for u in ORDER], [UNIT_COLOR[u] for u in ORDER]
    fig, axes = plt.subplots(4, 2, figsize=(14, 17.5),
                             gridspec_kw={"height_ratios": [3, 10, 10, 10]})
    for v, col in enumerate(axes.T):
        side = f"x_0..x_{t}" if v == 0 else f"x_0..x_{t - 1}"
        # values: affine least squares, weights scaled by unit std
        Wv = []
        for k in range(3):
            Hk, y = sel(H, Y[v, k, t])                         # states with k-th max > 0
            w, *_ = np.linalg.lstsq(np.c_[Hk, np.ones(len(Hk))], y, rcond=None)
            Wv.append(w[:-1] * Hk.std(0))
        heat(col[0], np.array(Wv), ul, NAMES, f"Value probes, window {side}: w_u·σ_u",
             xcolors=uc, xbounds=BOUNDS, fontsize=6)
        # ages: multinomial logistic regression on standardised units
        for k in range(3):
            Hk, y = sel(H, Y[v, 3 + k, t])
            Z, y = StandardScaler().fit_transform(Hk), y.astype(int)
            clf = LogisticRegression(C=10.0, max_iter=2000).fit(Z, y)
            heat(col[1 + k], clf.coef_, ul, [f"age {c}" for c in clf.classes_],
                 f"{AGE_NAMES[k]}, window {side}: W[age, u]",
                 xcolors=uc, xbounds=BOUNDS, fmt="{:.1f}", fontsize=5.5)
    for ax in axes[-1]:
        ax.set_xlabel("hidden unit")
    fig.suptitle(f"Linear-probe weights at t = {t} (probes read h_{t}; own colour scale per "
                 "panel, red > 0, blue < 0)\nTrained only on states where the k-th max > 0", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    group_legend(fig, y=0.0)
    fig.savefig(OUT / f"probe_weights_t{t}.png", dpi=170)
    print(f"wrote probe_weights_t{t}.png")


if __name__ == "__main__":
    main()
