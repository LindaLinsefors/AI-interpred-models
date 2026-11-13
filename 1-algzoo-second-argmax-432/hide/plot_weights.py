"""Visualise the weights of the 432-parameter AlgZoo second-argmax RNN (M_16,10).

The network: h_t = ReLU(W_ih x_t + W_hh h_(t-1)), h_(-1) = 0, logits = W_out h_9.
No biases. 16 + 256 + 160 = 432 parameters.

Weights are read (never modified) from the colleague's folder
../reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10_weights.npz.

Unit groups follow the paper in that folder:
  comparators 1, 6, 7, 8 | pulses 4, 5, 9 | wave units 0, 2, 3, 10-15.

Run:  python plot_weights.py      (needs numpy and matplotlib)
Writes PNGs to ../plots/ (extras to ../plots/more_plots/).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent.parent  # case-study folder
PLOTS = HERE / "plots"                         # figures go here; unused extras in plots/more_plots
WEIGHTS = (HERE.parent / "reverse-engineered-networks" / "1-algzoo-second-argmax-432"
           / "code" / "m16_10_weights.npz")

GROUPS = [
    ("comparators", [1, 6, 7, 8], "#2a78d6"),
    ("pulses", [4, 5, 9], "#d6582a"),
    ("wave units", [0, 2, 3, 10, 11, 12, 13, 14, 15], "#6b6b66"),
]
ORDER = [u for _, units, _ in GROUPS for u in units]
BOUNDS = np.cumsum([len(units) for _, units, _ in GROUPS])[:-1]
UNIT_COLOR = {u: c for _, units, c in GROUPS for u in units}

# Diverging blue <-> red with a neutral gray midpoint (negative = blue, positive = red).
CMAP = LinearSegmentedColormap.from_list(
    "bgr", ["#1f4e9c", "#6f9fd8", "#f0efec", "#e08a6c", "#a8321d"])
INK, MUTED = "#222220", "#8a8a85"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load():
    d = np.load(WEIGHTS)
    return d["W_ih"], d["W_hh"], d["W_out"]


def heat(ax, M, xlabels, ylabels, title, xcolors=None, ycolors=None,
         xbounds=(), ybounds=(), fmt="{:.2f}", vmax=None, fontsize=6.5):
    vmax = vmax or np.abs(M).max()
    im = ax.imshow(M, cmap=CMAP, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="equal")
    for (i, j), v in np.ndenumerate(M):
        strong = abs(v) > 0.55 * vmax
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                color="white" if strong else INK)
    ax.set_xticks(range(M.shape[1]), xlabels)
    ax.set_yticks(range(M.shape[0]), ylabels)
    ax.tick_params(length=0)
    for lbls, cols in ((ax.get_xticklabels(), xcolors), (ax.get_yticklabels(), ycolors)):
        if cols:
            for t, c in zip(lbls, cols):
                t.set_color(c)
                t.set_fontweight("bold")
    for b in xbounds:
        ax.axvline(b - 0.5, color=INK, lw=1.2)
    for b in ybounds:
        ax.axhline(b - 0.5, color=INK, lw=1.2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    return im


def group_legend(fig, y=0.015):
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, markersize=8,
                          label=f"{name}: units {', '.join(map(str, units))}")
               for name, units, c in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, y), fontsize=9)


def fig_overview(W_ih, W_hh, W_out, order, suffix, grouped):
    """All three matrices, same unit order on every hidden-unit axis."""
    ul = [str(u) for u in order]
    uc = [UNIT_COLOR[u] for u in order] if grouped else None
    bnd = BOUNDS if grouped else ()

    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 16, 16], wspace=0.18,
                          left=0.03, right=0.97, top=0.88, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    heat(ax, W_ih[order], ["x"], ul, "W_ih\n(input → unit)", ycolors=uc,
         ybounds=bnd, fmt="{:.2f}", fontsize=7)
    ax.set_ylabel("hidden unit")

    ax = fig.add_subplot(gs[1])
    im = heat(ax, W_hh[np.ix_(order, order)], ul, ul,
              "W_hh  (row = receiving unit at t, column = sending unit at t−1)",
              xcolors=uc, ycolors=uc, xbounds=bnd, ybounds=bnd)
    ax.set_xlabel("from unit (h_{t−1})")
    ax.set_ylabel("to unit (h_t)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    ax = fig.add_subplot(gs[2])
    im = heat(ax, W_out[:, order], ul, [f"pos {p}" for p in range(10)],
              "W_out  (unit at t=9 → logit for each position)",
              xcolors=uc, xbounds=bnd, fmt="{:.1f}")
    ax.set_xlabel("hidden unit (h_9)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, shrink=0.7)

    how = "units grouped by role" if grouped else "units in original order"
    fig.suptitle(f"M₁₆,₁₀ second-argmax RNN: all 432 weights ({how}). "
                 "Red = positive, blue = negative; each matrix has its own colour scale.",
                 x=0.03, ha="left", fontsize=12, color=INK)
    if grouped:
        group_legend(fig)
    fig.savefig((PLOTS if grouped else PLOTS / "more_plots") / f"weights_{suffix}.png", dpi=170)
    plt.close(fig)


def fig_eigen(W_hh):
    """Eigenvalues of W_hh: full 16x16 and the 13 non-pulse units (as in full_account.py)."""
    non_pulse = [u for u in range(16) if u not in (4, 5, 9)]
    sets = [("all 16 units", np.linalg.eigvals(W_hh), "#a8a8a3", 70, "o"),
            ("13 non-pulse units", np.linalg.eigvals(W_hh[np.ix_(non_pulse, non_pulse)]),
             "#2a78d6", 36, "D")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=0.8, ls="--")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5)
    for name, ev, c, s, m in sets:
        ax.scatter(ev.real, ev.imag, s=s, color=c, marker=m, label=name,
                   edgecolor="white", linewidth=1, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Eigenvalues of W_hh (dashed = unit circle)", loc="left", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Table of the non-pulse modes: radius and rotation per step.
    ev = sets[1][1]
    ev = ev[np.argsort(-np.abs(ev))]
    rows, seen = [], set()
    for e in ev:
        key = (round(e.real, 6), round(abs(e.imag), 6))
        if key in seen:
            continue
        seen.add(key)
        ang = np.degrees(abs(np.angle(e)))
        kind = "rotation (pair)" if abs(e.imag) > 1e-9 else ("real, flips sign" if e.real < 0 else "real")
        rows.append([f"{abs(e):.3f}", f"{ang:.2f}°", kind])
    ax = axes[1]
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["radius |λ|", "angle / step", "mode"],
                   loc="upper center", cellLoc="center", colWidths=[0.25, 0.25, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dcdcd8")
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Modes of the 13 non-pulse units (largest first).\n"
                 "The ~42° rotation is the paper's 'clock'.", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "more_plots" / "eigenvalues.png", dpi=170)
    plt.close(fig)


def fig_unit_profiles(W_ih, W_hh, W_out):
    """One small panel per unit: its input weight, incoming/outgoing recurrent weights, readout."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 13.6), sharey=False)
    lim_hh = np.abs(W_hh).max()
    for ax, u in zip(axes.flat, ORDER):
        c = UNIT_COLOR[u]
        x = np.arange(16)
        ax.bar(x - 0.2, W_hh[u, ORDER], width=0.4, color=c, label="in: W_hh[u, ·]")
        ax.bar(x + 0.2, W_hh[ORDER, u], width=0.4, color=c, alpha=0.35, label="out: W_hh[·, u]")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylim(-lim_hh * 1.05, lim_hh * 1.05)
        ax.set_xticks(x, [str(v) for v in ORDER], fontsize=6.5)
        for b in BOUNDS:
            ax.axvline(b - 0.5, color="#dcdcd8", lw=0.8)
        ax2 = ax.inset_axes([0.0, -0.62, 1.0, 0.32])
        ax2.bar(range(10), W_out[:, u], color=c, width=0.7)
        ax2.axhline(0, color=MUTED, lw=0.6)
        ax2.set_ylim(-8, 8)
        ax2.set_xticks(range(10), [str(p) for p in range(10)], fontsize=6.5)
        ax2.tick_params(labelsize=6.5)
        ax2.set_title("W_out[·, u] by output position", fontsize=7, loc="left", color=MUTED)
        ax.set_title(f"unit {u}   W_ih = {W_ih[u, 0]:+.2f}   self = {W_hh[u, u]:+.2f}",
                     fontsize=9.5, loc="left", color=c, fontweight="bold")
        for a in (ax, ax2):
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.5)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Per-unit weight profiles. Top: recurrent weights into the unit (solid) and "
                 "out of it (faint), senders/receivers in grouped order. Bottom: readout.",
                 x=0.02, ha="left", fontsize=11.5)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.115, hspace=1.05, wspace=0.18)
    group_legend(fig, y=0.935)
    fig.savefig(PLOTS / "more_plots" / "unit_profiles.png", dpi=150)
    plt.close(fig)


def main():
    W_ih, W_hh, W_out = load()
    assert W_ih.size + W_hh.size + W_out.size == 432
    fig_overview(W_ih, W_hh, W_out, ORDER, "grouped", grouped=True)
    print("wrote: plots/weights_grouped.png")


if __name__ == "__main__":
    main()
