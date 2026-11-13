"""Weights of the linear probes, per hidden unit. One probe per target, shared by all t
(fit by linear_probes.py and steps_classifier.py; read from value_probes.npz and
age_probes.npz).

  values (max, 2nd, 3rd):  y ~ w . h + b.  Shown as w_u * sigma_u (sigma_u = std of unit u
                           over the pooled training states): the change in the prediction
                           per one standard deviation of unit u.
  steps since (a_1..a_3):  softmax_c(W z + b), z = (h - mu) / sigma. Shown as W[c, u],
                           one row per class c (age). Softmax only sees differences between
                           rows; the L2 penalty picks the solution whose rows sum to ~0.

Units are ordered and coloured by group as in ../plot_weights.py (comparators,
pulses, wave units). Each panel has its own colour scale (red > 0, blue < 0).

Run:  python probe_weights.py      (run the two probe scripts first)
Writes probe_weights.png into OUT (../../<this folder's name>/).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from linear_probes import HIDE, NAMES, OUT, ROOT

sys.path.insert(0, str(ROOT / "hide"))                 # for plot_weights.py
from plot_weights import BOUNDS, INK, ORDER, UNIT_COLOR, group_legend, heat  # noqa: E402

AGE_NAMES = ["steps since max", "steps since 2nd", "steps since 3rd"]


def main():
    V, A = np.load(HIDE / "value_probes.npz"), np.load(HIDE / "age_probes.npz")
    ul, uc = [str(u) for u in ORDER], [UNIT_COLOR[u] for u in ORDER]
    fig, axes = plt.subplots(4, 2, figsize=(14, 17.5),
                             gridspec_kw={"height_ratios": [3, 10, 10, 10]})
    for v, col in enumerate(axes.T):
        side = "incl. x_t" if v == 0 else "excl. x_t"
        heat(col[0], (V["w"][v] * V["sigma"][v])[:, ORDER], ul, NAMES,
             f"Value probes ({side}): w_u·σ_u", xcolors=uc, xbounds=BOUNDS, fontsize=6)
        for k in range(3):
            i = 3 * v + k
            heat(col[1 + k], A[f"W{i}"][:, ORDER], ul, [f"age {c}" for c in A[f"classes{i}"]],
                 f"{AGE_NAMES[k]} ({side}): W[age, u]",
                 xcolors=uc, xbounds=BOUNDS, fmt="{:.1f}", fontsize=5.5)
    for ax in axes[-1]:
        ax.set_xlabel("hidden unit")
    fig.suptitle("Linear-probe weights, one probe per target shared by all t (own colour scale "
                 "per panel, red > 0, blue < 0)", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    group_legend(fig, y=0.0)
    fig.savefig(OUT / "probe_weights.png", dpi=170)
    print("wrote probe_weights.png")


if __name__ == "__main__":
    main()
