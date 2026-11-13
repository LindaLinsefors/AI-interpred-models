"""weights_grouped.png, but in the eigenmode basis of the full 16x16 W_hh (pulses not singled out).

Real block-diagonal basis: V has one column per real eigenvector and two (Re, Im) per complex
pair, each normalised so its largest entry is +1. With m = V^-1 h:
  mode pre-activation = V^-1 W_ih x + (V^-1 W_hh V) m ,   h = ReLU(V @ mode pre-activation)
  logits              = (W_out V) m
V^-1 W_hh V is block diagonal (pole, or 2x2 rotation per complex pair). The ReLU acts on units.

Run:  python plot_weights_eigenmodes.py   ->   ../plots/weights_eigenmodes.png
"""
import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np

from plot_weights import PLOTS, INK, heat, load

MODE_COLOR = "#3a8a5c"


def real_eigenbasis(W):
    """Columns of V and the list of mode groups, largest |eigenvalue| first."""
    ev, U = np.linalg.eig(W)
    order = np.argsort(-np.abs(ev), kind="stable")
    cols, groups = [], []
    for k in order:
        e, u = ev[k], U[:, k]
        if e.imag < -1e-12:
            continue  # take each conjugate pair once, from its +imag member
        u = u / u[np.argmax(np.abs(u))]  # largest entry -> exactly +1 (real)
        if abs(e.imag) <= 1e-12:
            groups.append((len(cols),))
            cols.append(u.real)
        else:
            groups.append((len(cols), len(cols) + 1))
            cols += [u.real, u.imag]
    return np.column_stack(cols), groups


def main():
    W_ih, W_hh, W_out = load()
    V, groups = real_eigenbasis(W_hh)
    Vinv = np.linalg.inv(V)
    W_ih_m, W_hh_m, W_out_m = Vinv @ W_ih, Vinv @ W_hh @ V, W_out @ V
    off = W_hh_m.copy()
    for g in groups:
        off[np.ix_(g, g)] = 0
    print(f"largest off-block entry of V^-1 W_hh V: {np.abs(off).max():.1e}")

    labels = [f"e{i}" for i in range(16)]
    colors = [MODE_COLOR] * 16
    bnd = [g[-1] + 1 for g in groups[:-1]]

    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 16, 16], wspace=0.18,
                          left=0.03, right=0.97, top=0.88, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    heat(ax, W_ih_m, ["x"], labels, "W_ih\n(input)", ycolors=colors, fmt="{:.2f}", fontsize=7)
    ax.set_ylabel("eigenmode")

    ax = fig.add_subplot(gs[1])
    im = heat(ax, W_hh_m, labels, labels,
              "W_hh  (row = receiving at t, column = sending at t−1)",
              xcolors=colors, ycolors=colors)
    for b in bnd:
        ax.axvline(b - 0.5, color=INK, lw=0.5, alpha=0.5)
        ax.axhline(b - 0.5, color=INK, lw=0.5, alpha=0.5)
    ax.set_xlabel("from mode (t−1)")
    ax.set_ylabel("to mode (t)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    ax = fig.add_subplot(gs[2])
    im = heat(ax, W_out_m, labels, [f"pos {p}" for p in range(10)],
              "W_out  (mode at t=9 → logit for each position)",
              xcolors=colors, xbounds=bnd, fmt="{:.1f}")
    ax.set_xlabel("eigenmode (t=9)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    fig.suptitle("M₁₆,₁₀ second-argmax RNN in the eigenmode basis of the full W_hh: "
                 "m = V⁻¹h, units = ReLU(V · mode pre-activation). "
                 "Each matrix has its own colour scale.",
                 x=0.03, ha="left", fontsize=11, color=INK)
    handles = [plt.Line2D([], [], marker="s", ls="", color=MODE_COLOR, markersize=8,
                          label="eigenmodes e0–e15 of W_hh, largest |λ| first "
                                "(thin lines: real modes / complex pairs)")]
    fig.legend(handles=handles, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=9)
    fig.savefig(PLOTS / "weights_eigenmodes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
