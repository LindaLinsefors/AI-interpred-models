"""Weights of the linear probes, per hidden unit. One probe per target, shared by all t
(positive-only variant: trained only on states where the k-th max > 0;
fit by linear_probes.py and steps_classifier.py; read from value_probes.npz and
age_probes.npz). The record-increment probe is the exception: it uses every state (fit by
increment_probe.py; read from increment_probe.npz).

  values (max, 2nd, 3rd):  y ~ w . h + b.  Shown as the raw w_u: the change in the
                           prediction per unit of h_u.
  record increment d_t:    d_t = ReLU(x_t - max(0, x_0..x_(t-1))) ~ w . h + b, also raw w_u.
  steps since (a_1..a_3):  softmax_c(W z + b), z = (h - mu) / sigma, which is linear in h
                           with weights W[c, u] / sigma_u. Shown as those raw weights, one
                           row per class c (age). Softmax only sees differences between
                           rows; the L2 penalty picks the solution whose rows sum to ~0.

Units are ordered and coloured by group as in ../plot_weights.py (comparators,
pulses, wave units). Each panel has its own colour scale (red > 0, blue < 0).

Run:  python probe_weights.py      (run the three probe scripts first)
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
    D = np.load(HIDE / "increment_probe.npz")
    ul, uc = [str(u) for u in ORDER], [UNIT_COLOR[u] for u in ORDER]
    fig = plt.figure(figsize=(14, 18.5))
    grid = fig.add_gridspec(5, 2, height_ratios=[3, 1, 10, 10, 10])
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in (0, 2, 3, 4)])
    mid = fig.add_subplot(grid[1, :])
    heat(mid, D["w"][ORDER][None], ul, ["d_t"],
         "Record increment d_t = ReLU(x_t − max(0, x_0..x_(t−1))): raw w_u; no states left out",
         xcolors=uc, xbounds=BOUNDS, fmt="{:.3f}", fontsize=5.5)
    for v, col in enumerate(axes.T):
        side = "incl. x_t" if v == 0 else "excl. x_t"
        heat(col[0], V["w"][v][:, ORDER], ul, NAMES,
             f"Value probes ({side}): raw w_u", xcolors=uc, xbounds=BOUNDS, fmt="{:.3f}", fontsize=5.5)
        for k in range(3):
            i = 3 * v + k
            heat(col[1 + k], (A[f"W{i}"] / A[f"sigma{i}"])[:, ORDER], ul,
                 [f"age {c}" for c in A[f"classes{i}"]],
                 f"{AGE_NAMES[k]} ({side}): raw weight on h_u",
                 xcolors=uc, xbounds=BOUNDS, fmt="{:.2f}", fontsize=5.5)
    for ax in axes[-1]:
        ax.set_xlabel("hidden unit")
    fig.suptitle("Linear-probe weights, one probe per target shared by all t (own colour scale "
                 "per panel, red > 0, blue < 0)\nPositive-only variant: trained only on "
                 "states where the k-th max > 0 (except d_t)", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    a, b, m = axes[0, 0].get_position(), axes[0, 1].get_position(), mid.get_position()
    mid.set_position([(a.x0 + b.x1 - a.width) / 2, (m.y0 + m.y1 - a.height / 3) / 2,
                      a.width, a.height / 3])              # centred, same cell size as the value probes
    group_legend(fig, y=0.0)
    fig.savefig(OUT / "probe_weights.png", dpi=170)
    print("wrote probe_weights.png")


if __name__ == "__main__":
    main()
