"""Three figures for sections 7 and 4.2 of the report: where M_16,10 fails, and the wave units' low points.

Run:  python -B plot_errors_units.py            (recomputes, about 35 s, caches plot_errors_units.npz)
      python -B plot_errors_units.py --redraw   (only redraws from the cache)
Both print the values to compare with the report. Writes to ../plots/:

errors_by_arrangement.png
    Maximum at the last position. P(answer = position of the third largest) for every placement of the runner-up
    (rows) and the third (columns), for (S - T)/S in 3-8% and in 8-20%, third above 0.2 (the selection of
    e22_separation_errors.py). The inputs are N(0,1) conditioned exactly on "the maximum is last": the order
    statistics of iid inputs are independent of where they sit, so sorted draws are placed with the largest at
    position 9 and the others in random order. 8M draws give about 10,000 and 22,000 sequences per cell.
accuracy_by_gaps.png
    2^23 N(0,1) sequences (the rarest cell of panel a then holds about 600).
    (a) Accuracy over the two relative differences (M - S+)/M and (S+ - T+)/M, for sequences with a positive
    runner-up, in seven colour classes. (b) Share of all errors in the regions that these differences define
    (the classes of section 7). (c) Accuracy against the number of positive inputs.
wave_unit_troughs.png
    500,000 N(0,1) sequences. Median of h_t[u] / M_t by age of the maximum, for the nine wave units and the four
    comparators, over states with M_t > 0.3 and a runner-up at least one step old. Each row is coloured relative to
    its own largest value. Cells where the unit is exactly zero in more than 80% of those states are outlined.
    The title of (a) also names the units whose lowest median (ages 1-9) is below 5% of the row's largest without
    being outlined. The footnote adds how small the previous maximum is at age 0, and how often all four
    comparators are off at a new maximum above a positive earlier one (the selection of the report's 94%).
"""
import sys

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import *
from fig_mechanism import style, SURF, INK, INK2, MUTED, AXIS, BLUE

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_errors_units.npz"
REL_EDGES = np.array([0.0, 0.03, 0.08, 0.20, 0.50])                      # bins of (S - T)/S used in section 7
GAP_EDGES = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])   # bins of both relative differences
ACC_STEPS = np.array([1, 0.999, 0.99, 0.95, 0.9, 0.75, 0.5, 0])          # colour classes of the accuracy heatmap
QUOTED = [(0, 7), (8, 0), (7, 0), (7, 8), (2, 1), (6, 5), (7, 6)]        # (p_S, p_T) of the arrangements named in section 7


# ---------------------------------------------------------------- data
def answer(x):
    """The network's answer (position of the largest logit) for inputs x (B, 10), without storing the states."""
    h = np.zeros((len(x), N))
    for t in range(x.shape[1]):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
    return (h @ U.T).argmax(1)


def compute():
    out = {}

    # figure 1: maximum at position 9; sequences by (bin of (S - T)/S, p_S, p_T), and how many answer p_T
    rng = np.random.default_rng(71)
    n, third = np.zeros(4 * 81), np.zeros(4 * 81)
    for _ in range(8):
        v = -np.sort(-rng.standard_normal((1_000_000, 10)), axis=1)      # values, largest first
        rel = (v[:, 1] - v[:, 2]) / v[:, 1]
        keep = (v[:, 2] > 0.2) & (rel <= REL_EDGES[-1])
        v, rel = v[keep], rel[keep]
        place = np.argsort(rng.random((len(v), 9)), axis=1)             # positions of the 2nd..10th largest
        x = np.empty_like(v)
        x[:, 9] = v[:, 0]
        np.put_along_axis(x[:, :9], place, v[:, 1:], axis=1)
        cell = (np.searchsorted(REL_EDGES[1:], rel) * 9 + place[:, 0]) * 9 + place[:, 1]
        n += np.bincount(cell, minlength=324)
        third += np.bincount(cell, weights=answer(x) == place[:, 1], minlength=324)
    out["arr_n"], out["arr_third"] = n.reshape(4, 9, 9), third.reshape(4, 9, 9)

    # figure 2: sequences and correct answers by the two relative differences (S > 0), and by number of positive inputs
    nb = len(GAP_EDGES) - 1
    n, right, pos_n, pos_right, easy = np.zeros(nb * nb), np.zeros(nb * nb), np.zeros(11), np.zeros(11), np.zeros(2)
    for seed in range(8):
        x = sample(2 ** 20, 7700 + seed)
        ok = answer(x) == second_argmax(x)
        k = (x > 0).sum(1)
        pos_n += np.bincount(k, minlength=11)
        pos_right += np.bincount(k, weights=ok, minlength=11)
        v, ok = -np.sort(-x[k >= 2], axis=1), ok[k >= 2]                 # two positive inputs: positive runner-up
        M, S, T3 = v[:, 0], v[:, 1], np.maximum(v[:, 2], 0)
        b1 = np.minimum(np.searchsorted(GAP_EDGES, (M - S) / M, side="right") - 1, nb - 1)
        b2 = np.minimum(np.searchsorted(GAP_EDGES, (S - T3) / M, side="right") - 1, nb - 1)
        n += np.bincount(b2 * nb + b1, minlength=nb * nb)
        right += np.bincount(b2 * nb + b1, weights=ok, minlength=nb * nb)
        e = (M - S > 0.2 * M) & (S - T3 > 0.2 * M) & (v[:, 2] > 0)       # both differences above 0.2 M, positive third
        easy += e.sum(), ok[e].sum()
    out.update(gap_n=n.reshape(nb, nb), gap_right=right.reshape(nb, nb), pos_n=pos_n, pos_right=pos_right, easy=easy)

    # figure 3: median of h/M and share of exact zeros, by unit and age of the maximum
    x = sample(500_000, 4301)
    H, st = run(x), running_stats(x)
    sel = (st["M1"] > 0.3) & (st["a2"] >= 1)
    med, zero = np.empty((N, 10)), np.empty((N, 10))
    for a in range(10):
        s = sel & (st["a1"] == a)
        med[:, a] = np.median(H[s] / st["M1"][s][:, None], axis=0)
        zero[:, a] = (H[s] == 0).mean(0)
        if a == 0:
            out["all_off"] = (H[s][:, COMP] == 0).all(1).mean()          # all four comparators off as the maximum arrives
            out["prev_max"] = np.median(np.maximum(st["M2"][s], 0) / st["M1"][s])    # previous maximum / new maximum
    new = (st["a1"][:, 1:] == 0) & (st["M1"][:, :-1] > 0)                # the report's selection: above a positive maximum
    out["all_off_new"] = (H[:, 1:][new][:, COMP] == 0).all(1).mean()
    out.update(med=med, zero=zero)
    np.savez(CACHE, **out)
    return out


# ---------------------------------------------------------------- drawing helpers
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ.set_bad(SURF)


def axes_in(fig, left, bottom, width, height):
    """Axes placed in inches."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def header(fig, title, sub):
    fw, fh = fig.get_size_inches()
    fig.text(0.35 / fw, 1 - 0.3 / fh, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.35 / fw, 1 - 0.62 / fh, sub, fontsize=9, color=INK2, va="top", linespacing=1.4)


def footnote(fig, s):
    fw, fh = fig.get_size_inches()
    fig.text(0.35 / fw, 0.2 / fh, s, fontsize=8.5, color=INK2, va="bottom", linespacing=1.4)


def tiles(ax, A, norm, title, **kw):
    """Heatmap of A as tiles separated by a gap in the surface colour; NaN cells stay empty."""
    style(ax, title)
    im = ax.imshow(A, cmap=SEQ, norm=norm, aspect="auto", **kw)
    ax.grid(False)
    ax.autoscale(False)
    for k in range(1, A.shape[1]):
        ax.axvline(k - 0.5, color=SURF, lw=2.5)
    for k in range(1, A.shape[0]):
        ax.axhline(k - 0.5, color=SURF, lw=2.5)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def write(ax, im, i, j, s, dy=0.0, **kw):
    """Text in cell (row i, column j): white on a dark fill, ink on a light one."""
    r, g, b = np.array(im.cmap(im.norm(im.get_array()[i, j]))[:3]) ** 2.2
    dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.2
    ax.text(j, i + dy, s, ha="center", va="center", color="#ffffff" if dark else INK, **{"fontsize": 8.5, **kw})


def outline(ax, i, j):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.5, zorder=3, clip_on=False))


def colourbar(fig, im, rect, label):
    cb = fig.colorbar(im, cax=axes_in(fig, *rect))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    cb.set_label(label, color=INK2, fontsize=9)
    return cb


def rounded_bar(ax, x, y, width, height):
    """Thin bar with rounded corners. Start it outside the axes so that the baseline end is clipped square."""
    fig, bb = ax.figure, ax.get_position()
    in_x = bb.width * fig.get_figwidth() / np.ptp(ax.get_xlim())                # inches per x unit
    in_y = bb.height * fig.get_figheight() / np.ptp(ax.get_ylim())
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={0.03 / in_x}",
                                mutation_aspect=in_x / in_y, fc=BLUE, ec="none"))


def percent(a):
    """Accuracy in %, with as many decimals as its distance from 100 needs."""
    e = 100 * (1 - a)
    return f"{100 * a:.{0 if e >= 10 or e == 0 else 1 if e >= 1 else 2 if e >= 0.01 else 3}f}"


# ---------------------------------------------------------------- figure 1
def draw_arrangements(d):
    rate = d["arr_third"] / np.where(d["arr_n"] > 0, d["arr_n"], np.nan)          # (bin, p_S, p_T); NaN on the diagonal
    overall = d["arr_third"].sum((1, 2)) / d["arr_n"].sum((1, 2))
    per_cell = [round(d["arr_n"][b].sum() / 72, -2) for b in (1, 2)]
    fig = plt.figure(figsize=(13.7, 7.9))
    header(fig, "Maximum at the last position: how often the third largest is named instead of the runner-up",
           "P(answer = position of the third largest) for every placement of the runner-up (rows) and the third (columns), "
           "by how far the third's value T is below the runner-up's S, as (S − T) / S.\nInputs: N(0,1), conditioned on "
           f"the maximum being at position 9 and T > 0.2; about {per_cell[0]:,.0f} sequences per cell in (a) and "
           f"{per_cell[1]:,.0f} in (b).  Arrangements are written (position of runner-up, of third, of maximum).")
    for k, (b, name) in enumerate([(1, "3–8%"), (2, "8–20%")]):
        R = rate[b]
        i0, j0 = np.unravel_index(np.nanargmax(R), R.shape)
        title = (f"{'ab'[k]}  Third {name} below the runner-up: named in {overall[b]:.0%} of cases, "
                 f"in {R[i0, j0]:.0%} for ({i0}, {j0}, 9)")
        ax = axes_in(fig, 0.75 + 6.1 * k, 1.1, 5.4, 5.4)
        im = tiles(ax, R, Normalize(0, 0.8), title)
        for i, j in zip(*np.nonzero(R >= 0.05)):
            q = (i, j) in QUOTED
            write(ax, im, i, j, f"{R[i, j]:.2f}", dy=-0.14 * q)
            if q:
                write(ax, im, i, j, f"({i}, {j}, 9)", dy=0.2, fontsize=6.8)
                outline(ax, i, j)
        ax.set_xticks(range(9)); ax.set_yticks(range(9))
        ax.set_xlabel("position of the third largest"); ax.set_ylabel("position of the runner-up")
    cb = colourbar(fig, im, (12.5, 2.2, 0.16, 3.2), "P(answer = position of the third largest)")
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    footnote(fig, "Numbers: cells with P ≥ 0.05.  Outlined: the arrangements named in section 7 of the report.\n"
                  "Row 8: the runner-up arrived on the step just before the maximum; column 8: the third did.  "
                  "Empty diagonal: not possible.")
    fig.savefig(OUT / "errors_by_arrangement.png", dpi=150)
    print("wrote", OUT / "errors_by_arrangement.png")
    print("  P(third) in the four bins of (S - T)/S, max at position 9:")
    for i, j in QUOTED:
        print(f"    ({i}, {j}, 9): " + "  ".join(f"{r:.3f}" for r in rate[:, i, j]))
    print("    all 72 arrangements: " + "  ".join(f"{r:.4f}" for r in overall))


# ---------------------------------------------------------------- figure 2
def draw_gaps(d):
    n, right, pos_n, pos_right = d["gap_n"], d["gap_right"], d["pos_n"], d["pos_right"]
    total, errors = pos_n.sum(), (pos_n - pos_right).sum()
    acc = right / np.where(n > 0, n, np.nan)                              # rows: (S+ - T+)/M, columns: (M - S+)/M
    lo = np.arange(len(n)) < 4                                            # the bins below 0.05

    def block(rows, cols):                                                # sequences and errors in a block of the grid
        s = np.ix_(rows, cols)
        return n[s].sum(), (n - right)[s].sum()
    regions = [("Runner-up close to third:  S⁺ − T⁺ < 0.05 M ≤ M − S⁺", *block(lo, ~lo)),
               ("Maximum close to runner-up:  M − S⁺ < 0.05 M ≤ S⁺ − T⁺", *block(~lo, lo)),
               ("Both differences below 0.05 M", *block(lo, lo)),
               ("Neither:  both differences at least 0.05 M", *block(~lo, ~lo)),
               ("Runner-up not positive:  S ≤ 0", pos_n[:2].sum(), (pos_n - pos_right)[:2].sum())]
    close = sum(r[2] for r in regions[:3]) / errors
    neither = 1 - regions[3][2] / regions[3][1]
    not_positive = regions[4][2] / errors

    fig = plt.figure(figsize=(14.4, 8.45))
    header(fig, "Accuracy by how close the three largest inputs are to each other",
           "M ≥ S ≥ T are the three largest of the ten inputs, S⁺ = max(S, 0), T⁺ = max(T, 0).  "
           f"N(0,1) inputs, {total:,.0f} sequences, overall accuracy {pos_right.sum() / total:.2%}.")

    # (a) heatmap over the two relative differences
    ax = axes_in(fig, 1.35, 1.6, 5.76, 5.76)
    im = tiles(ax, 1 - acc, BoundaryNorm(1 - ACC_STEPS, SEQ.N),
               f"a  Accuracy is {neither:.1%} when both differences exceed 0.05 M", origin="lower")
    for i, j in zip(*np.nonzero(n > 0)):
        write(ax, im, i, j, percent(acc[i, j]))
    ax.text(7, 7, "not\npossible", ha="center", va="center", fontsize=7.5, color=MUTED)
    for line in (ax.axvline, ax.axhline):
        line(3.5, color=SURF, lw=8)                                       # wider gap at 0.05
    ticks = [f"{e:g}" for e in GAP_EDGES]
    ax.set_xticks(np.arange(9) - 0.5, ticks); ax.set_yticks(np.arange(9) - 0.5, ticks)
    kw = dict(color=INK2, fontsize=9, clip_on=False)
    ax.plot([-0.45, 3.4], [-1.02, -1.02], color=AXIS, lw=1, clip_on=False)
    ax.text(1.5, -1.1, "maximum close to runner-up", ha="center", va="top", **kw)
    ax.plot([-1.22, -1.22], [-0.45, 3.4], color=AXIS, lw=1, clip_on=False)
    ax.text(-1.3, 1.5, "runner-up close to third", ha="right", va="center", rotation=90, **kw)
    ax.text(3.5, -1.55, "(M − S⁺) / M:  maximum minus runner-up, relative to the maximum", ha="center", va="top", **kw)
    ax.text(-1.78, 3.5, "(S⁺ − T⁺) / M:  runner-up minus third, relative to the maximum", ha="right", va="center",
            rotation=90, **kw)
    cb = colourbar(fig, im, (7.3, 2.9, 0.16, 3.2), "accuracy, %")
    cb.set_ticks(1 - ACC_STEPS, labels=[f"{100 * a:g}" for a in ACC_STEPS])
    cb.minorticks_off(); cb.ax.invert_yaxis()

    # (b) share of all errors by region
    ax = axes_in(fig, 9.05, 4.75, 5.0, 2.61)
    style(ax, f"b  {close:.0%} of all errors: two of the top three within 0.05 M of each other")
    ax.grid(False); ax.set_xlim(0, 100); ax.set_ylim(-0.45, 4.75)
    ax.set_xticks([]); ax.set_yticks([]); ax.spines["bottom"].set_visible(False)
    for y, (name, seqs, errs) in zip(range(4, -1, -1), sorted(regions, key=lambda r: -r[2])):
        rounded_bar(ax, -5, y - 0.14, 5 + 100 * errs / errors, 0.28)
        ax.text(100 * errs / errors + 1.5, y, f"{errs / errors:.1%} of errors", va="center", fontsize=8.5, color=INK)
        ax.text(0, y + 0.27, name, va="bottom", fontsize=8.5, color=INK2)
        ax.text(100, y + 0.27, f"{seqs / total:.1%} of inputs", va="bottom", ha="right", fontsize=8.5, color=MUTED)

    # (c) accuracy by number of positive inputs
    ax = axes_in(fig, 9.05, 1.6, 5.0, 2.2)
    style(ax, "c  With fewer than two positive inputs the runner-up is rarely found")
    ax.xaxis.grid(False); ax.set_xlim(-0.6, 10.6); ax.set_ylim(0, 1.12)
    a = pos_right / pos_n
    for k in range(11):
        rounded_bar(ax, k - 0.17, -0.2, 0.34, a[k] + 0.2)
        ax.text(k, a[k] + 0.025, f"{a[k]:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(k, -0.125, f"{pos_n[k] / total:.1%}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=MUTED)
    ax.plot([-0.35, 1.35], [0.34, 0.34], color=AXIS, lw=1)
    ax.text(0.5, 0.37, f"runner-up\nnot positive:\n{not_positive:.1%} of errors", ha="center", va="bottom", fontsize=8.5,
            color=INK2, linespacing=1.25)
    ax.set_xticks(range(11)); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("accuracy")
    ax.set_xlabel("number of positive inputs   (grey: share of all inputs)", labelpad=24)

    easy_n, easy_right = d["easy"]
    footnote(fig, f"(a) covers the sequences with a positive runner-up ({1 - pos_n[:2].sum() / total:.1%} of inputs); "
                  "bins are log-spaced, the first one starts at 0, and a wider gap marks 0.05.\nBoth differences above "
                  f"0.2 M and a positive third ({easy_n / total:.1%} of inputs): accuracy {easy_right / easy_n:.2%}, "
                  f"that is {easy_n - easy_right:,.0f} errors in {easy_n:,.0f} sequences.")
    fig.savefig(OUT / "accuracy_by_gaps.png", dpi=150)
    print("wrote", OUT / "accuracy_by_gaps.png")
    print(f"  accuracy {pos_right.sum() / total:.4f} on {total:.0f} sequences")
    for name, seqs, errs in regions:
        print(f"    {name:58s} share of inputs {seqs / total:.4f}  accuracy {1 - errs / seqs:.4f}  "
              f"share of errors {errs / errors:.4f}")
    print(f"    both differences above 0.2 M and T > 0: accuracy {easy_right / easy_n:.5f}, share {easy_n / total:.4f}")
    print("    accuracy by number of positive inputs 0..10: " + " ".join(f"{v:.3f}" for v in a))
    print("    smallest number of sequences in a cell of (a):", int(n[n > 0].min()))


# ---------------------------------------------------------------- figure 3
def draw_troughs(d):
    med, zero = d["med"], d["zero"]
    rel = med / med.max(1, keepdims=True)
    low = rel[:, 1:].argmin(1) + 1                                        # age (1..9) at which the unit is lowest
    wave = sorted(WAVE, key=lambda u: (low[u], rel[u, low[u]]))
    marked = zero > 0.8
    zw = sorted((a, u) for u in WAVE for a in range(10) if marked[u, a])
    near = sorted((low[u], u) for u in WAVE if rel[u, low[u]] < 0.05 and not marked[u, low[u]])   # deep, not outlined
    names = lambda pairs, k: ", ".join(str(p[k]) for p in pairs)
    title_a = f"a  Wave units {names(zw, 1)} are exactly zero at ages {names(zw, 0)} of the maximum"
    if near:
        dips = ", ".join(sorted({f"{med[u, a]:.1f}" for a, u in near}))
        title_a += f"; the medians of units {names(near, 1)} fall to {dips} M at ages {names(near, 0)}"
    fig = plt.figure(figsize=(11.0, 9.35))
    header(fig, "Wave units and comparators: activation by the age of the maximum",
           "Median of h[u] / M over states with M > 0.3 and a runner-up at least one step old (M: largest input so far; "
           "N(0,1) inputs, 500,000 sequences).\nColour: the median relative to the largest value in its row.  "
           "Numbers: the medians.")
    titles = [title_a,
              f"b  Comparators: all four are off in {d['all_off']:.0%} of the steps at which the maximum arrives; "
              "afterwards they swing with its age"]
    for units, bottom, title in zip((wave, COMP), (4.0, 1.4), titles):
        ax = axes_in(fig, 1.0, bottom, 8.0, 0.45 * len(units))
        im = tiles(ax, rel[units], Normalize(0, 1), title)
        for i, u in enumerate(units):
            for a in range(10):
                if marked[u, a]:
                    write(ax, im, i, a, f"zero in\n{zero[u, a]:.1%}", fontsize=7.2, linespacing=1.15)
                    outline(ax, i, a)
                else:
                    write(ax, im, i, a, f"{med[u, a]:.1f}", fontsize=8)
        ax.set_xticks(range(10)); ax.set_yticks(range(len(units)), [f"unit {u}" for u in units])
    ax.set_xlabel("age of the maximum: steps since it arrived (0: it is the current input)")
    cb = colourbar(fig, im, (9.25, 4.7, 0.16, 2.6), "median h[u] / M, relative to the largest in its row")
    cb.set_ticks([0, 0.5, 1])
    footnote(fig, "Wave-unit rows are sorted by the age (1–9) at which the unit is lowest.  Outlined: the unit is exactly "
                  "zero in more than 80% of these states (share printed).\nAge 0: the maximum is the current input, and "
                  "the wave units are still set mostly by the earlier inputs (the previous maximum is "
                  f"{d['prev_max']:.2f} M in the median), so h[u] / M is small there.\n(b): the {d['all_off']:.0%} holds "
                  "for these states; over all new maxima above a positive earlier maximum, the selection of the report, "
                  f"all four are off in {d['all_off_new']:.0%}.")
    fig.savefig(OUT / "wave_unit_troughs.png", dpi=150)
    print("wrote", OUT / "wave_unit_troughs.png")
    print("  cells that are exactly zero in more than 80% of states (unit, age, share): "
          + ", ".join(f"({u}, {a}, {zero[u, a]:.3f})" for u in WAVE + COMP for a in range(10) if marked[u, a]))
    print(f"  all four comparators exactly zero at age 0: {d['all_off']:.3f}; at a new maximum above a positive earlier "
          f"maximum: {d['all_off_new']:.3f}; median of max(previous maximum, 0) / M at age 0: {d['prev_max']:.3f}")
    print("  median h[2]/M at ages 0..9: " + " ".join(f"{v:.2f}" for v in med[2]))
    print(f"  unit 12: median at age 4 {med[12, 4]:.2f}, exactly zero in {zero[12, 4]:.3f}; "
          f"largest share of zeros of units 3 and 15: {zero[3].max():.3f}, {zero[15].max():.3f}")


if __name__ == "__main__":
    data = dict(np.load(CACHE)) if ("--redraw" in sys.argv and CACHE.exists()) else compute()
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF})
    OUT.mkdir(exist_ok=True)
    draw_arrangements(data)
    draw_gaps(data)
    draw_troughs(data)
