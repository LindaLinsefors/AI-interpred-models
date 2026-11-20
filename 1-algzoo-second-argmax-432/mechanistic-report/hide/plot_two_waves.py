"""Figures for sections 3.4 and 4.3 of report.md: the two cancelling waves.

The scores s = U h_9 are split into V (the readout through the comparators 1, 6, 7, 8) and m (the rest).

  ../plots/waves_by_age.png        Coefficient of the gap M - S in V, in m and in s = V + m at t = 9, by age of the
                                   maximum (rows) and logit index 0..7 (columns).  States with all four comparators
                                   on and S > 0.1; regression on (M - S, S, x_9) without intercept, fitted
                                   separately for each age (as e15_two_score_vectors.py).
  ../plots/readout_columns.png     Left: the comparators' readout columns U[i, j], with sinusoids at the clock's
                                   frequency fitted on i = 0..7 (the legend also gives the best frequency and R2
                                   when the frequency is fitted too).  Right: amplitude of the clock pattern
                                   2 Re(z r_clock) in each logit, read through the comparators, through the nine
                                   wave units and through all 13 (as e07_clock_readout_split.py).  Weights only.
  ../plots/margin_by_distance.png  Median and quartiles of s[p_S] - s[p_M] and of its V part and m part, in units
                                   of M, against D = |p_M - p_S|, for four bins of the relative gap (as
                                   e16_m_scores.py: t = 9, all four comparators on, S > 0.2, both positions <= 8).
                                   Dashed: medians of the two parts without the pairs that include position 8 (an
                                   input one step old, off the wave of indices 0..7); these pairs are half of the
                                   states at D = 7 and all at D = 8, so the pooled medians there mix two kinds.

Run:  python -B plot_two_waves.py            (1M N(0,1) sequences, about 5 s; caches plot_two_waves.npz)
      python -B plot_two_waves.py --redraw   (only redraws from the cache)
Both print the numbers to compare with the report.
"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from common import COMP, ORD, U, WAVE, run, sample, w
from fig_mechanism import AQUA, AXIS, BLUE, DIV, INK, INK2, MUTED, ORANGE, SURF, YELLOW, legend, style
from modes import LAM, RIGHT

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_two_waves.npz"
OMEGA = np.angle(LAM["clock"])                               # 42.75 degrees per step, in radians
AGES = np.arange(2, 10)                                      # age of the maximum at t = 9; its index is 9 - age
GAPS = [(0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.80)]   # bins of the relative gap (M - S) / M
GREYS = ["#aeaca4", "#898781", "#52514e", INK]               # ordinal ramp for those bins, small to large
MK = dict(ms=7.5, mec=SURF, mew=1.6)
DASH = (0, (4, 2))                                           # line style of the medians without position 8
NOTE = dict(fontsize=8.5, color=INK2, linespacing=1.5,       # notes inside a panel: surface colour behind the text
            bbox=dict(facecolor=SURF, edgecolor="none", pad=2))
PM, PS = r"p$_\mathrm{M}$", r"p$_\mathrm{S}$"


def compute(n=1_000_000, seed=3415, chunk=200_000):
    x = sample(n, seed)
    h9, on = np.empty((n, 16)), np.empty(n, bool)            # final state; are all four comparators on at t = 9?
    for i in range(0, n, chunk):
        H, Z = run(x[i:i + chunk], return_pre=True)
        h9[i:i + chunk], on[i:i + chunk] = H[:, -1], (Z[:, -1][:, COMP] > 0).all(1)
    ar = np.arange(n)
    pM, pS = np.argsort(-x, axis=1)[:, :2].T
    M, S = x[ar, pM], x[ar, pS]
    V = h9[:, COMP] @ U[:, COMP].T
    s = h9 @ U.T
    m = s - V

    # waves_by_age: coefficient of M - S in V and in m, per age of the maximum and logit index 0..7
    coef = np.empty((3, 8, 8))
    for k, a in enumerate(AGES):
        q = on & (S > 0.1) & (pM == 9 - a)
        X = np.column_stack([M[q] - S[q], S[q], x[q, 9]])
        for j, Y in enumerate((V, m)):
            coef[j, k] = np.linalg.lstsq(X, Y[q, :8], rcond=None)[0][0]
    coef[2] = coef[0] + coef[1]                              # the regression is linear, so this is the fit for s

    # margin_by_distance: quartiles of the runner-up's margin over the maximum and of its two parts
    ok = on & (S > 0.2) & (pM < 9) & (pS < 9)
    old = (pM < 8) & (pS < 8)                                # neither is at position 8: both at least two steps old
    gap, D = (M - S) / M, np.abs(pM - pS)
    part = np.stack([(Y[ar, pS] - Y[ar, pM]) / M for Y in (V, m, s)])
    quart = np.empty((len(GAPS), 8, 3, 3))                   # gap bin, D - 1, (25%, 50%, 75%), (V, m, s)
    med_old = np.full((len(GAPS), 8, 3), np.nan)             # medians without the pairs that include position 8
    count = np.empty((len(GAPS), 8), int)
    for b, (lo, hi) in enumerate(GAPS):
        for d in range(1, 9):
            q = ok & (D == d) & (gap > lo) & (gap <= hi)
            quart[b, d - 1], count[b, d - 1] = np.percentile(part[:, q], [25, 50, 75], axis=1), q.sum()
            if d < 8:                                        # D = 8 is always the pair of positions 0 and 8
                med_old[b, d - 1] = np.median(part[:, q & old], axis=1)
    binned = ok & (gap > GAPS[0][0]) & (gap <= GAPS[-1][1])
    share8 = np.array([1 - old[binned & (D == d)].mean() for d in range(1, 9)])   # share of pairs with position 8
    out = dict(coef=coef, quart=quart, count=count, med_old=med_old, share8=share8, n=n)
    np.savez(CACHE, **out)
    return out


def sinusoid_fits(om=OMEGA):
    """U[i, j] ~ u_j cos(om i - phi_j) + offset_j on i = 0..7: returns u, phi, offset, R2 for the comparators."""
    i = np.arange(8)
    B = np.column_stack([np.cos(om * i), np.sin(om * i), np.ones(8)])
    Y = U[:8, COMP]
    c = np.linalg.lstsq(B, Y, rcond=None)[0]
    r2 = 1 - ((Y - B @ c) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return np.hypot(c[0], c[1]), np.arctan2(c[1], c[0]), c[2], r2


def free_frequency(grid=np.radians(np.arange(30, 60, 0.01))):
    """The same fit with the frequency fitted too: best frequency (degrees per index) and its R2, per comparator."""
    r2 = np.array([sinusoid_fits(om)[3] for om in grid])
    return np.degrees(grid[r2.argmax(0)]), r2.max(0)


def clock_readout():
    """Complex readout of the clock pattern per unit of z: through the comparators and through the wave units."""
    r, Uo = RIGHT["clock"], U[:, ORD]
    ci, wi = [ORD.index(u) for u in COMP], [ORD.index(u) for u in WAVE]
    return 2 * Uo[:, ci] @ r[ci], 2 * Uo[:, wi] @ r[wi]


def ratio(a, b, sl):
    """Size of the sum of two readouts relative to their two sizes added, over the indices sl."""
    return np.linalg.norm((a + b)[sl]) / (np.linalg.norm(a[sl]) + np.linalg.norm(b[sl]))


def signed(v, digits=0, space=""):
    """Number with an explicit sign and a typographic minus, e.g. +65, −33, or '+ 0.51' with space=' '."""
    return ("−" if v < 0 else "+") + space + f"{abs(v):.{digits}f}"


def subtitle(fig, x, text):
    fig.text(x, 0.975, text, fontsize=9.5, color=INK2, va="top", linespacing=1.5)


def opaque(lg):
    """Give a legend the surface colour as background, so that no gridline runs through its text."""
    lg.set_frame_on(True); lg.get_frame().set(facecolor=SURF, edgecolor="none", alpha=1)
    return lg


def fig_waves(d):
    C = d["coef"]
    vmax = np.abs(C).max()
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.45))
    fig.subplots_adjust(left=0.04, right=0.925, top=0.835, bottom=0.135, wspace=0.1)
    titles = ["a  V, read through the comparators: a trough on the maximum",
              "b  m, read through the other units: a crest on the maximum",
              f"c  Their sum s = V + m: at most {np.abs(C[2]).max():.0f} logits per unit of gap are left"]
    for k, ax in enumerate(axs):
        style(ax, titles[k])
        ax.grid(False)
        im = ax.pcolormesh(np.arange(9) - 0.5, np.arange(9) + 1.5, C[k], cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax),
                           edgecolors=SURF, linewidth=2)
        for r, a in enumerate(AGES):
            for i in range(8):
                v = C[k, r, i]
                ax.text(i, a, signed(v) if abs(v) >= 0.5 else "0", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if abs(v) > 0.72 * vmax else INK)
            ax.add_patch(Rectangle((9 - a - 0.5, a - 0.5), 1, 1, fill=False, ec=INK, lw=1.4, clip_on=False))
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 7.5); ax.set_ylim(9.5, 1.5)
        ax.set_xticks(range(8)); ax.set_yticks(AGES)
        ax.set_xlabel("logit index i (the input's position)")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_ylabel("age of the maximum (steps)")
    cax = fig.add_axes([0.94, 0.2, 0.011, 0.59])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    cb.set_label("logits per unit of M − S", color=INK2, fontsize=9)
    subtitle(fig, 0.04, "Coefficient of the gap M − S (largest minus second-largest input) in the final scores "
             "s = U h₉ = V + m, by age of the maximum and logit index.  V: the part read through the comparators "
             "(units 1, 6, 7, 8); m: the rest.\nStates at t = 9 with all four comparators on and S > 0.1 "
             f"(inputs iid N(0,1), {int(d['n']):,} sequences); regression on M − S, S and x₉ without intercept, fitted "
             "separately for each age.")
    own = C[:, np.arange(8), 9 - AGES].mean(1)               # mean over the eight outlined cells of V, m and the sum
    fig.legend(handles=[Patch(fill=False, ec=INK, lw=1.4, label="the maximum's own index, 9 − age   (mean of these "
                              f"eight cells: V {signed(own[0])}, m {signed(own[1])}, sum {signed(own[2])})")],
               loc="lower left", bbox_to_anchor=(0.035, 0.015), frameon=False, fontsize=9, labelcolor=INK2,
               handlelength=1.1, handleheight=1.1)
    fig.savefig(OUT / "waves_by_age.png", dpi=150)
    plt.close(fig)


def fig_readout():
    u, phi, off, r2 = sinusoid_fits()
    fo, fr2 = free_frequency()
    Rc, Rw = clock_readout()
    dphi = np.abs(np.degrees(np.angle(Rw[:8] / Rc[:8])))
    lo7, all10 = ratio(Rc, Rw, slice(0, 8)), ratio(Rc, Rw, slice(0, 10))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 6.25))
    fig.subplots_adjust(left=0.05, right=0.975, top=0.835, bottom=0.1, wspace=0.17)
    subtitle(fig, 0.05, "From the weights alone.  U[i, j]: readout weight from unit j to logit i; the comparators are "
             "units 1, 6, 7, 8.\nClock pattern: 2 Re(z r$_\\mathrm{clock}$), the clock mode's share of the state of the "
             "13 non-pulse units; z is its coordinate, scaled so that a pulse of unit 4 adds its size to z "
             f"({w[4]:.3f} per unit of input excess over θ₄).")
    idx = np.arange(10)

    style(ax, "a  For i ≤ 7 the comparators' readout columns are close to sinusoids at the clock's frequency")
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(7.5, color=AXIS, lw=1)
    fine, beyond = np.linspace(0, 7, 200), np.linspace(7, 9, 60)
    for k, (j, col) in enumerate(zip(COMP, [BLUE, ORANGE, AQUA, YELLOW])):
        f = lambda i: u[k] * np.cos(OMEGA * i - phi[k]) + off[k]
        ax.plot(fine, f(fine), "-", color=col, lw=2)
        ax.plot(beyond, f(beyond), "-", color=col, lw=1, alpha=0.5)
        ax.plot(idx, U[:, j], "o", color=col, **MK,
                label=f"unit {j}:  {u[k]:.2f} cos({np.degrees(OMEGA):.2f}° i {signed(-np.degrees(phi[k]), 0, ' ')}°) "
                      f"{signed(off[k], 2, ' ')},   R² = {r2[k]:.3f}   ({fo[k]:.1f}°, R² = {fr2[k]:.3f})")
        ax.text(-0.22, f(0), f"unit {j}", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(7.58, -8.6, "not fitted", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xticks(idx); ax.set_xlim(-1.15, 9.4); ax.set_ylim(-9, 15)
    ax.set_xlabel("logit index i"); ax.set_ylabel("readout weight U[i, j] of comparator j")
    lg = opaque(legend(ax, loc="upper left", alignment="left",
                       title="points: U[i, j];   curves: clock frequency fixed; amplitude, phase and offset fitted on "
                             "i = 0..7\nin brackets: frequency per index and R² when the frequency is fitted too"))
    lg.get_title().set_color(INK2); lg.get_title().set_fontsize(8.5)

    style(bx, f"b  The clock pattern's two readouts cancel: {lo7:.1%} of their summed sizes is left")
    for R, col, lab, name in ((Rw, ORANGE, "through the nine wave units (the wave in m)", "wave units"),
                              (Rc, BLUE, "through the comparators (the wave in V)", "comparators"),
                              (Rc + Rw, INK, "sum: through all 13 non-pulse units", "sum")):
        bx.plot(idx, np.abs(R), "-o", color=col, lw=2, label=lab, **MK)
        bx.text(9.2, np.abs(R[9]), name, va="center", fontsize=8.5, color=INK2)
    bx.text(0.03, 0.33, f"indices 0–7: the two parts are {dphi.min():.0f}°–{dphi.max():.0f}° out of phase,\n"
            f"and |sum| / (|comparators| + |wave units|) = {lo7:.3f}\n"
            f"({all10:.4f} over all ten indices;  |·| = norm over the indices)",
            transform=bx.transAxes, va="bottom", **NOTE)
    bx.set_xticks(idx); bx.set_xlim(-0.4, 10.6); bx.set_ylim(0, 12.5)
    bx.set_xlabel("logit index i"); bx.set_ylabel("amplitude in logit i per unit of clock coordinate |z|")
    opaque(legend(bx, loc="upper left"))
    fig.savefig(OUT / "readout_columns.png", dpi=150)
    plt.close(fig)


def fig_margin(d):
    Q = d["quart"]                                           # gap bin, D - 1, quartile, (V, m, s)
    Ds = np.arange(1, 9)
    fig = plt.figure(figsize=(16.5, 6.25))
    top, bottom = 0.835, 0.11
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.24, 1.9], left=0.045, right=0.985, top=top, bottom=bottom,
                          wspace=0.08)
    subtitle(fig, 0.045, "Scores s = U h₉ = V + m, with V the part read through the comparators (units 1, 6, 7, 8) and m "
             f"the rest.  M, S: largest and second-largest input, at positions {PM}, {PS}.  Relative gap: (M − S)/M.\n"
             "States at t = 9 with all four comparators on, S > 0.2 and both positions older than the last input "
             f"(inputs iid N(0,1), {int(d['n']):,} sequences).  Lines: medians; bands: interquartile range; dashed: "
             "medians without the pairs that include position 8.")
    series = ((0, BLUE, f"V part: V[{PS}] − V[{PM}]"), (1, ORANGE, f"m part: m[{PS}] − m[{PM}]"),
              (2, INK, f"sum: the margin s[{PS}] − s[{PM}]"))
    for b, (lo, hi) in enumerate(GAPS):
        ax = fig.add_subplot(gs[b])
        style(ax, "a  The margin is the small remainder of two parts that grow with the gap" if b == 0 else "")
        ax.axhline(0, color=AXIS, lw=1)
        for p, col, lab in series:
            ax.fill_between(Ds, Q[b, :, 0, p], Q[b, :, 2, p], color=col, alpha=0.12, lw=0)
            ax.plot(Ds, Q[b, :, 1, p], "-o", color=col, lw=2, label=lab, ms=6, mec=SURF, mew=1.3)
            if p < 2:                                        # the two parts again, without the pairs at position 8
                ax.plot(Ds[:7], d["med_old"][b, :7, p], color=col, lw=1.3, ls=DASH, zorder=1.9)
        ax.text(0.5, 0.965, f"relative gap {lo:.2f}–{hi:.2f}", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=INK2)
        ax.set_xticks(Ds); ax.set_xlim(0.5, 8.5); ax.set_ylim(-100, 125)
        if b:
            ax.tick_params(labelleft=False)
    axs = fig.axes
    axs[0].set_ylabel("score at the runner-up minus score at the maximum, in units of M")
    hs, ls = axs[0].get_legend_handles_labels()
    opaque(legend(axs[0], handles=hs + [Patch(color=MUTED, alpha=0.25, lw=0),
                                        Line2D([], [], color=MUTED, lw=1.3, ls=DASH)], loc="lower left",
                  labels=ls + ["bands: 25%–75% of states", "dashed: V part and m part\nwithout position 8"]))
    lab = dict(ha="center", fontsize=8.5, color=INK2)        # direct labels in the last panel, beside D = 6
    axs[3].text(6.45, Q[3, 5, 2, 0] + 4, "V part", va="bottom", **lab)
    axs[3].text(5.6, Q[3, 4:6, 2, 2].max() + 3, "margin", va="bottom", **lab)
    axs[3].text(6.45, Q[3, 5, 0, 1] - 4, "m part", va="top", **lab)
    fig.text(0.5 * (axs[0].get_position().x0 + axs[3].get_position().x1), bottom - 0.075 * (top - bottom),
             f"D = |{PM} − {PS}|, distance between maximum and runner-up (positions)", ha="center", va="top",
             fontsize=9, color=INK2)

    bx = fig.add_subplot(gs[5])
    style(bx, "b  The margin depends on the distance more than on the gap")
    names = [f"{lo:.2f}–{hi:.2f}" for lo, hi in GAPS]
    for b in range(len(GAPS)):
        bx.plot(Ds, Q[b, :, 1, 2], "-o", color=GREYS[b], lw=2, label=names[b], **MK)
    ends = Q[:, -1, 1, 2]                                    # direct labels at D = 8, pushed apart where they collide
    y = np.sort(ends)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + 1.0)
    for b, yy in zip(np.argsort(ends), y):
        bx.text(8.25, yy, names[b], va="center", fontsize=8.5, color=INK2)
    bx.set_xticks(Ds); bx.set_xlim(0.5, 9.75); bx.set_ylim(0, y[-1] + 1.5)
    bx.set_xlabel(f"D = |{PM} − {PS}|"); bx.xaxis.set_label_coords(0.5, -0.075)
    bx.set_ylabel(f"median margin s[{PS}] − s[{PM}], in units of M")
    lg = opaque(legend(bx, loc="upper left", title="relative gap (M − S)/M"))
    lg.get_title().set_color(INK2); lg.get_title().set_fontsize(8.5)
    sh, mo = d["share8"], d["med_old"][:, 6, 2]              # share of pairs with position 8; margins at D = 7 without
    bx.text(0.985, 0.03, f"Position 8 (an input one step old) is in {100 * sh[:6].min():.0f}–{sh[:6].max():.0%} of the "
            f"pairs at D ≤ 6,\nin {sh[6]:.0%} at D = 7 and in all at D = 8.  Without those pairs the median\nmargin at D = 7 is "
            f"{mo.min():.0f}–{mo.max():.0f}, against {Q[:, 6, 1, 2].min():.0f}–{Q[:, 6, 1, 2].max():.0f} for all pairs.",
            transform=bx.transAxes, ha="right", va="bottom", multialignment="left", **NOTE)
    fig.savefig(OUT / "margin_by_distance.png", dpi=150)
    plt.close(fig)


def report_numbers(d):
    """Print the values to compare with report.md (sections 3.4 and 4.3)."""
    C, Q = d["coef"], d["quart"]
    delta = np.subtract.outer(AGES - 9, -np.arange(8))       # index - (9 - age)
    byd = np.array([[C[k][delta == dd].mean() for dd in range(-7, 8)] for k in range(3)])
    print("coefficient of M - S by distance from the maximum's index, -7..7 (report: table of section 3.4)")
    for k, name in enumerate(("V", "m", "sum")):
        print(f"  {name:3s}", " ".join(f"{v:+6.1f}" for v in byd[k]))
    dd = np.arange(-7, 8)
    B = np.column_stack([np.cos(OMEGA * dd), np.sin(OMEGA * dd), np.ones(15)])
    for k, name in enumerate(("V", "m")):
        c = np.linalg.lstsq(B, byd[k], rcond=None)[0]
        ext = np.degrees(np.arctan2(c[1], c[0])) / np.degrees(OMEGA)
        print(f"  cosine fitted to the {name} row: amplitude {np.hypot(c[0], c[1]):.1f}, crest at distance {ext:+.2f} "
              f"(trough half a period away: {ext - np.sign(ext) * np.pi / OMEGA:+.2f})")
    print(f"  largest |coefficient| per cell: V {np.abs(C[0]).max():.1f}, m {np.abs(C[1]).max():.1f}, sum {np.abs(C[2]).max():.1f}")
    u, phi, off, r2 = sinusoid_fits()
    print("readout columns of units 1, 6, 7, 8 (report: u = 5.81, 5.09, 6.21, 6.08; phi = -29, 26, 131, -130; R2 = 0.974, 0.943, 0.991, 0.988)")
    print("  u", np.round(u, 2), " phi", np.round(np.degrees(phi), 1), " offset", np.round(off, 2), " R2", np.round(r2, 3))
    fo, fr2 = free_frequency()
    print("  frequency fitted too (verifier V2: 48.65, 49.05, 44.35, 44.90 degrees):", np.round(fo, 2), " R2", np.round(fr2, 3))
    Rc, Rw = clock_readout()
    print("clock pattern in the logits, indices 0..9 (report: 5.6-8.4, 5.7-9.5 and 0.4-1.6 on 0..7; ratio 5.5%, 5.75% over all ten)")
    for name, R in (("comparators", Rc), ("wave units", Rw), ("sum", Rc + Rw)):
        print(f"  {name:11s}", " ".join(f"{abs(v):5.2f}@{np.degrees(np.angle(v)):+5.0f}" for v in R))
    print(f"  ratio over 0..7: {ratio(Rc, Rw, slice(0, 8)):.4f};  over 0..9: {ratio(Rc, Rw, slice(0, 10)):.4f}")
    print("  phase difference on 0..7:", np.round(np.abs(np.degrees(np.angle(Rw[:8] / Rc[:8]))), 1))
    print("margins in units of M: median V part, m part, sum for D = 1..8 (report: table of section 3.4);")
    print("'old': the same without the pairs that include position 8 (not in the report)")
    for b, (lo, hi) in enumerate(GAPS):
        print(f"  relative gap {lo:.2f}-{hi:.2f}")
        for p, name in enumerate(("V part", "m part", "sum")):
            print(f"    {name:6s}", " ".join(f"{v:6.1f}" for v in Q[b, :, 1, p]),
                  "   old", " ".join(f"{v:6.1f}" for v in d["med_old"][b, :7, p]))
        print("    n     ", " ".join(f"{v:6d}" for v in d["count"][b]))
    print("  share of the pairs that include position 8, D = 1..8:", np.round(d["share8"], 3))


if __name__ == "__main__":
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF,
                         "mathtext.fontset": "custom", "mathtext.rm": "Segoe UI"})
    OUT.mkdir(exist_ok=True)
    data = dict(np.load(CACHE)) if ("--redraw" in sys.argv and CACHE.exists()) else {}
    if "med_old" not in data:                                # no cache, or one written before the split by position 8
        data = compute()
    report_numbers(data)
    fig_waves(data); fig_readout(); fig_margin(data)
    print("wrote waves_by_age.png, readout_columns.png, margin_by_distance.png in", OUT)
