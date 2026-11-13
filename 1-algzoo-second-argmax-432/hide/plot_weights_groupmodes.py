"""weights_grouped.png, with comparators and wave units each replaced by their own internal eigenmodes.

Basis (block diagonal in neuron space, groups never mixed):
  comparators 1,6,7,8  -> eigenmodes c0-c3 of W_hh[comp, comp]
  pulses 4,5,9         -> kept as individual neurons
  wave units           -> eigenmodes w0-w8 of W_hh[wave, wave]
Each group uses the real block-diagonal eigenbasis from plot_weights_eigenmodes.py. With h = V m:
  mode pre-activation = V^-1 W_ih x + (V^-1 W_hh V) m ,   h = ReLU(V @ mode pre-activation)
  logits              = (W_out V) m
The diagonal blocks of V^-1 W_hh V are block diagonal; the blocks between groups are not.

Exception: a complex pair whose Im vector is much shorter than its Re vector (nearly real, e.g.
comparator pair 0.041 +/- 0.013i) gives a badly scaled basis. For such pairs the two columns are
replaced by an orthogonal basis of the same invariant plane (each scaled so its largest entry is +1),
so their 2x2 block is no longer a pure rotation (it becomes nearly upper-triangular).

Run:  python plot_weights_groupmodes.py   ->   ../plots/weights_groupmodes.png
"""
import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np

from plot_weights import GROUPS, PLOTS, INK, heat, load
from plot_weights_eigenmodes import real_eigenbasis

PREFIX = {"comparators": "c", "wave units": "w"}
FLAT = 0.2  # Im/Re length ratio below which a complex pair is treated as (nearly) real


def group_basis(B):
    V, groups = real_eigenbasis(B)
    for g in groups:
        if len(g) == 2 and np.linalg.norm(V[:, g[1]]) < FLAT * np.linalg.norm(V[:, g[0]]):
            Q, _ = np.linalg.qr(V[:, g])
            V[:, g] = Q / Q[np.abs(Q).argmax(0), range(2)]
            print(f"pair {g}: replaced by an orthogonal basis of its plane")
    return V, groups


def main():
    W_ih, W_hh, W_out = load()
    V = np.zeros((16, 16))
    labels, colors, bnd, thin = [], [], [], []
    for name, units, color in GROUPS:
        n0 = len(labels)
        if name in PREFIX:
            B, groups = group_basis(W_hh[np.ix_(units, units)])
            V[np.ix_(units, range(n0, n0 + len(units)))] = B
            labels += [f"{PREFIX[name]}{i}" for i in range(len(units))]
            thin += [n0 + g[-1] + 1 for g in groups[:-1]]
        else:
            for k, u in enumerate(units):
                V[u, n0 + k] = 1.0
            labels += [str(u) for u in units]
        colors += [color] * len(units)
        bnd.append(len(labels))
    bnd = bnd[:-1]

    Vinv = np.linalg.inv(V)
    W_ih_m, W_hh_m, W_out_m = Vinv @ W_ih, Vinv @ W_hh @ V, W_out @ V

    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 16, 16], wspace=0.18,
                          left=0.03, right=0.97, top=0.88, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    heat(ax, W_ih_m, ["x"], labels, "W_ih\n(input)", ycolors=colors,
         ybounds=bnd, fmt="{:.2f}", fontsize=7)
    ax.set_ylabel("mode / pulse unit")

    ax = fig.add_subplot(gs[1])
    im = heat(ax, W_hh_m, labels, labels,
              "W_hh  (row = receiving at t, column = sending at t−1)",
              xcolors=colors, ycolors=colors, xbounds=bnd, ybounds=bnd, vmax=2)  # colours clip at ±2
    for b in thin:
        ax.axvline(b - 0.5, color=INK, lw=0.5, alpha=0.5)
        ax.axhline(b - 0.5, color=INK, lw=0.5, alpha=0.5)
    ax.set_xlabel("from (t−1)")
    ax.set_ylabel("to (t)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7, extend='both')

    ax = fig.add_subplot(gs[2])
    im = heat(ax, W_out_m, labels, [f"pos {p}" for p in range(10)],
              "W_out  (mode/pulse at t=9 → logit for each position)",
              xcolors=colors, xbounds=bnd, fmt="{:.1f}")
    ax.set_xlabel("mode / pulse unit (t=9)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    fig.suptitle("M₁₆,₁₀ second-argmax RNN with per-group modes: comparators and wave units each "
                 "replaced by the eigenmodes of their own W_hh block; pulses kept as neurons. "
                 "Each matrix has its own colour scale.",
                 x=0.03, ha="left", fontsize=10.5, color=INK)
    text = {"comparators": "comparator modes c0–c3 (of units 1, 6, 7, 8)",
            "pulses": "pulses: units 4, 5, 9",
            "wave units": "wave modes w0–w8 (of units 0, 2, 3, 10–15)"}
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, markersize=8, label=text[name])
               for name, _, c in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=9)
    fig.savefig(PLOTS / "weights_groupmodes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
