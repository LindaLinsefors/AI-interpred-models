"""weights_grouped.png, but with the 13 non-pulse units replaced by the 13 modes of full_account.py.

Change of basis (exact, uses the trained weights): z = [m ; p], with
  m = ANALYSIS @ h[ORDINARY]   (13 modes)      p = h[PULSE_UNITS]   (3 pulses)
so h[ORDINARY] = SYNTHESIS @ m. Then
  mode pre-activation  = A W_ih[o] x + A W_oo S m + A W_op p ,   h[o] = ReLU(S @ mode pre-activation)
  pulse pre-activation =   W_ih[p] x +   W_po S m +   W_pp p ,   p    = ReLU(pulse pre-activation)
  logits               = W_out[:, o] S m + W_out[:, p] p
The ReLU still acts on units, not modes.

Run:  python plot_weights_modes.py   ->   ../plots/weights_modes.png
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_weights import HERE, PLOTS, INK, GROUPS, heat, load

sys.path.insert(0, str(HERE.parent / "reverse-engineered-networks" / "1-algzoo-second-argmax-432" / "code"))
import full_account as fa

MODE_COLOR = "#3a8a5c"
PULSE_COLOR = next(c for name, _, c in GROUPS if name == "pulses")


def to_modes(W_ih, W_hh, W_out):
    o, p = fa.ORDINARY, fa.PULSE_UNITS
    S, A = fa.SYNTHESIS, fa.ANALYSIS
    W_ih_m = np.vstack([A @ W_ih[o], W_ih[p]])
    W_hh_m = np.block([[A @ W_hh[np.ix_(o, o)] @ S, A @ W_hh[np.ix_(o, p)]],
                       [W_hh[np.ix_(p, o)] @ S, W_hh[np.ix_(p, p)]]])
    W_out_m = np.hstack([W_out[:, o] @ S, W_out[:, p]])
    return W_ih_m, W_hh_m, W_out_m


def main():
    W_ih, W_hh, W_out = to_modes(*load())
    labels = [f"m{i}" for i in range(13)] + [str(u) for u in fa.PULSE_UNITS]
    colors = [MODE_COLOR] * 13 + [PULSE_COLOR] * 3
    bnd = [13]
    mode_bnd = [g[-1] + 1 for g in fa.GROUPS[:-1]]  # thin lines between rotation groups

    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 16, 16], wspace=0.18,
                          left=0.03, right=0.97, top=0.88, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    heat(ax, W_ih, ["x"], labels, "W_ih\n(input)", ycolors=colors,
         ybounds=bnd, fmt="{:.2f}", fontsize=7)
    ax.set_ylabel("mode / pulse unit")

    ax = fig.add_subplot(gs[1])
    im = heat(ax, W_hh, labels, labels,
              "W_hh  (row = receiving at t, column = sending at t−1)",
              xcolors=colors, ycolors=colors, xbounds=bnd, ybounds=bnd)
    for b in mode_bnd:
        ax.plot([b - 0.5] * 2, [-0.5, 12.5], color=INK, lw=0.5, alpha=0.5)
        ax.plot([-0.5, 12.5], [b - 0.5] * 2, color=INK, lw=0.5, alpha=0.5)
    ax.set_xlabel("from (t−1)")
    ax.set_ylabel("to (t)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    ax = fig.add_subplot(gs[2])
    im = heat(ax, W_out, labels, [f"pos {p}" for p in range(10)],
              "W_out  (mode/pulse at t=9 → logit for each position)",
              xcolors=colors, xbounds=bnd, fmt="{:.1f}")
    ax.set_xlabel("mode / pulse unit (t=9)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    fig.suptitle("M₁₆,₁₀ second-argmax RNN in the mode basis: non-pulse units replaced by modes "
                 "m = ANALYSIS·h, units = ReLU(SYNTHESIS · mode pre-activation). "
                 "Each matrix has its own colour scale.",
                 x=0.03, ha="left", fontsize=11, color=INK)
    handles = [plt.Line2D([], [], marker="s", ls="", color=MODE_COLOR, markersize=8,
                          label="modes m0–m12 (thin lines: rotation groups)"),
               plt.Line2D([], [], marker="s", ls="", color=PULSE_COLOR, markersize=8,
                          label=f"pulses: units {', '.join(map(str, fa.PULSE_UNITS))}")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=9)
    fig.savefig(PLOTS / "weights_modes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
