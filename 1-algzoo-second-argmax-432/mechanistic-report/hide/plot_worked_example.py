"""Two figures for the worked example of report section 5.5,
x = (0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4).

worked_example_timeline.png: four stacked panels on the step axis t = 0..9.
    a  The input x_t, the five thresholds it meets (theta_1, theta_6, theta_7, theta_8, theta_4, all linear in
       h_{t-1}; a hollow marker = threshold below the input, so that comparator is off), and the max and
       runner-up so far as step lines. The input markers are drawn DODGE = 0.17 of a step to the right of
       their step, so that they do not cover the threshold markers (a new runner-up lies among them).
    b  The pulses h_t[4], h_t[5], h_t[9].
    c  The provisional scores s_t = U h_t. Row = input (position p), column = step t; the entry shown is
       s_t[p + 9 - t] minus the mean over the inputs before x_t at that step. (The mean over all entries
       would move a whole column by up to 20 logits whenever the newest input is low.) Frame = largest
       entry of the column; M, S = the true largest and second largest input so far.
    d  The same for m_t = s_t minus the comparators' part. Frame = largest entry among the inputs before x_t.
worked_example_clock.png: the clock coordinate z / w_4 after every step in the complex plane, on the phase
    law phi(a) of section 4.1 (rays), which is fitted here on sampled N(0,1) sequences. Hollow points show
    the two steps with a new runner-up as they would be for a low input (x_t = -1). The table beside the
    plot lists the plotted values.

Run:  python -B plot_worked_example.py            (fits the phase law on 500k sequences, about 10 s)
      python -B plot_worked_example.py --redraw   (reuses plot_worked_example.npz)
Both print the table of section 5.5 for comparison with the report.
"""
import sys

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from common import *
from modes import coords
from fig_mechanism import style, legend, SURF, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE, AQUA, YELLOW, RAMP, DIV

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_worked_example.npz"

X = np.array([0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4])
STEPS = np.arange(10)
THR = COMP + [4]                                       # the units whose thresholds x_t meets at once
# Comparators as in mechanism_summary.png. Unit 4 is INK2, not MUTED: theta_4 runs beside theta_7 here, and
# MUTED against AQUA fails the colour-blind check (OKLab distance 5.1 under deuteranopia; INK2 passes).
THR_COLOR = dict(zip(THR, [BLUE, ORANGE, AQUA, YELLOW, INK2]))
EVENT_MARK = {"new max": "D", "new runner-up": "s", "low": "o"}
VMAX = 50                                              # colour range of the score panels, in logits
DODGE = 0.17                                           # panel a: the input markers sit this far right of their step
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sub = lambda name, j: f"{name}{j}".translate(SUB)      # sub("θ", 7) = "θ₇"
minus = lambda s: s.replace("-", "−")


# ---------------------------------------------------------------- data
def example(x=X):
    """Everything the figures show about one sequence, computed from the weights."""
    H, Z = (a[0] for a in run(x, return_pre=True))
    Hprev = np.vstack([np.zeros(N), H[:-1]])
    st = {k: v[0] for k, v in running_stats(x[None]).items()}
    Mprev = np.r_[-np.inf, st["M1"][:-1]]
    Sprev = np.r_[0, np.maximum(st["M2"][:-1], 0)]             # S+ of the earlier inputs
    event = np.where(x > Mprev, "new max", np.where(x > Sprev, "new runner-up", "low"))
    s = H @ U.T                                                # s[t, index]
    m = s - H[:, COMP] @ U[:, COMP].T                          # without the comparators' part V
    P = STEPS[:, None]
    by_pos = lambda a: np.where(P <= STEPS, a[STEPS, (P + 9 - STEPS).clip(0, 9)], np.nan)   # [p, t]
    # the same step if the new runner-up had been a low input (x_t = -1)
    ghost = {int(t): coords(run(np.where(STEPS == t, -1.0, x))[0, t])["clock"] / w[4]
             for t in STEPS[event == "new runner-up"]}
    return dict(H=H, off=Z[:, COMP] <= 0, theta=-(Hprev @ W[THR].T) / w[THR], M=st["M1"], S=st["M2"],
                p1=st["p1"], p2=st["p2"], event=event, s=by_pos(s), m=by_pos(m), z=coords(H)["clock"] / w[4],
                ghost=ghost, lap=np.cumsum(event == "new max") - 1,
                answer=int((U @ H[-1]).argmax()), truth=int(second_argmax(x[None])[0]))


def phase_law(n=500_000, seed=55):
    """Section 4.1: per age a = 1..9 of the max, complex least squares
    z / w_4 ~ alpha(a) e^{i phi(a)} (M - S+) + beta(a) S+   on states with M > 0.  Returns alpha e^{i phi}."""
    x = sample(n, seed)
    st = running_stats(x)
    M, Sp, a1 = st["M1"], np.maximum(st["M2"], 0), st["a1"]
    z = coords(run(x))["clock"] / w[4]
    out = []
    for a in range(1, 10):
        k = (a1 == a) & (M > 0)
        A = np.column_stack([M[k] - Sp[k], Sp[k]]).astype(complex)
        out.append(np.linalg.lstsq(A, z[k], rcond=None)[0][0])
    return np.array(out)


def older(R):
    """Scores [p, t] of the inputs before x_t only (the entry of x_t itself, on the diagonal, is left out)."""
    return np.where(np.eye(10, dtype=bool), np.nan, R)


def tops(d):
    """Position of the largest entry of s_t, and of m_t among the inputs before x_t (-1 if there is none)."""
    return np.nanargmax(d["s"], 0), np.r_[-1, np.nanargmax(older(d["m"])[:, 1:], 0)]


def check(d):
    """The statements made in the panel titles and notes, verified for this sequence."""
    top_s, top_m = tops(d)
    H, new_max, new_ru = d["H"], d["event"] == "new max", d["event"] == "new runner-up"
    assert ((H[:, 4] > 0) == new_max).all() and d["off"][new_max].all()                 # unit 4; all four comparators off
    assert ((H[1:, 9] > 0) == new_max[:-1]).all() and ((H[1:, 5] > 0) == (new_max | new_ru)[:-1]).all()
    # panel a, steps 1..9: each threshold's position between the runner-up (0; zero if negative) and the max (1) of the earlier inputs
    Mp, Sp = d["M"][:-1, None], np.maximum(d["S"][:-1, None], 0)
    f = dict(zip(THR, ((d["theta"][1:] - Sp) / (Mp - Sp)).T))
    assert np.abs(d["theta"][1:, [THR.index(4), THR.index(7)]] - Mp).max() < 0.1                 # theta_4, theta_7 at the max
    assert f[8].max() < 0.5 and all(f[j].min() < 0.25 and f[j].max() > 0.75 for j in (1, 6))    # theta_8 low; theta_1, theta_6 swing
    assert (top_s[2:] == d["p2"][2:]).all() and (top_m[1:] == d["p1"][:-1]).all()
    # clock figure: near 0 deg one step after a new max, then forward; a new runner-up shortens the clock and holds it back
    arg = np.degrees(np.angle(d["z"]))
    assert np.abs(arg[1:][new_max[:-1]]).max() < 15 and (np.diff(arg)[~new_max[1:] & ~new_max[:-1]] > 0).all()
    assert all(abs(d["z"][t]) < abs(g) and arg[t] < np.degrees(np.angle(g)) for t, g in d["ghost"].items())


def print_table(d, law):
    top_s, top_m = tops(d)
    deg = lambda c: np.degrees(np.angle(c))
    print(f"x = {X}   second largest at position {d['truth']}, network answers {d['answer']}")
    print(" t   x_t  event          th4   th1   th6   th7   th8  off    h4     h5     h9    |z|/w4 @ deg   top s  top m")
    for t in STEPS:
        th = dict(zip(THR, d["theta"][t]))
        off = "".join(str(j) for j, o in zip(COMP, d["off"][t]) if o) or "-"
        print(f" {t}  {X[t]:5.2f}  {d['event'][t]:13s} {th[4]:5.2f} {th[1]:5.2f} {th[6]:5.2f} {th[7]:5.2f} {th[8]:5.2f}  "
              f"{off:5s} {d['H'][t, 4]:6.2f} {d['H'][t, 5]:6.2f} {d['H'][t, 9]:6.2f}   "
              f"{abs(d['z'][t]):5.3f} @ {deg(d['z'][t]):6.1f}   {top_s[t]:3d}   {top_m[t]:3d}")
    print("without the new runner-up (x_t = -1):", {t: f"{abs(g):.3f} @ {deg(g):.1f}" for t, g in d["ghost"].items()})
    print("phase law phi(a), a = 1..9:", np.round(deg(law) % 360, 1), "  alpha(a):", np.round(np.abs(law), 3))


# ---------------------------------------------------------------- drawing
def spread(y, gap):
    """Move label heights apart until neighbours are at least `gap` apart (order kept)."""
    order = np.argsort(y)
    v = np.asarray(y, float)[order]
    for _ in range(200):
        k = np.flatnonzero(np.diff(v) < gap - 1e-9)
        if not len(k):
            break
        shift = (gap - (v[k[0] + 1] - v[k[0]])) / 2
        v[k[0]] -= shift
        v[k[0] + 1] += shift
    out = np.empty_like(v)
    out[order] = v
    return out


def header(fig, d, title, y, dy):
    """Figure title and, under it, the sequence and the answer."""
    seq = minus(", ".join(f"{v:.1f}" for v in X))
    fig.text(0.012, y, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, y - dy, f"x = ({seq}).  The second largest input is {X[d['truth']]:.1f} at position {d['truth']}; "
             f"the network answers {d['answer']}.", fontsize=9.5, color=INK2, va="top")


def titled_legend(ax, handles, title, **kw):
    lg = legend(ax, handles=handles, title=title, **kw)
    lg.get_title().set_color(INK); lg.get_title().set_fontsize(8.5)
    lg._legend_box.align = "left"
    ax.add_artist(lg)
    ax.legend_ = None                                   # kept as an added artist only; as legend_ too it would be drawn twice


def draw_timeline(d):
    fig = plt.figure(figsize=(11, 15.2))
    header(fig, d, "One sequence, step by step (section 5.5)", y=0.985, dy=0.016)
    outer = fig.add_gridspec(4, 1, height_ratios=[4.3, 2.3, 3.1, 3.1], left=0.095, right=0.795, top=0.936,
                             bottom=0.044, hspace=0.27)
    xlim = (-0.5, 9.5)
    note = dict(fontsize=8.5, color=INK2)

    # (a) input, thresholds, max and runner-up so far
    ax = fig.add_subplot(outer[0])
    style(ax, f"a  {sub('θ', 4)} and {sub('θ', 7)} sit at the max of the earlier inputs, {sub('θ', 8)} near the runner-up; "
              f"{sub('θ', 1)} and {sub('θ', 6)} swing between the two")
    edges = np.r_[STEPS, 9.5]
    M, S = d["M"], np.where(np.isfinite(d["S"]), d["S"], np.nan)
    ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(S, 2), np.repeat(M, 2), color=INK, alpha=0.06, lw=0)
    ax.stairs(M, edges, baseline=None, color=MUTED, lw=1.2)
    ax.stairs(S[1:], edges[1:], baseline=None, color=MUTED, lw=1.2)
    ax.text(9.58, M[-1], "max so far", va="center", **note)
    ax.text(9.58, S[-1], "runner-up so far", va="center", **note)
    for k, j in enumerate(THR):
        th, c = d["theta"][:, k], THR_COLOR[j]
        ax.plot(STEPS, th, "-", color=c, lw=2, zorder=3)
        if j != 4:
            off = d["off"][:, k]
            ax.plot(STEPS[~off], th[~off], "o", color=c, ms=7, mec=SURF, mew=1.4, zorder=4)
            ax.plot(STEPS[off], th[off], "o", mfc=SURF, mec=c, ms=6, mew=1.6, zorder=4)
    for j, y in zip(THR, spread(d["theta"][-1], 0.11)):
        ax.text(9.15, y, sub("θ", j), va="center", **note)
    for ev, mk in EVENT_MARK.items():                           # just right of the step, clear of the threshold markers
        k = d["event"] == ev
        ax.plot(STEPS[k] + DODGE, X[k], mk, color=INK, ms=7.5 if mk == "D" else 8, mec=SURF, mew=1.6, zorder=6)
    ax.axhspan(2.008, 2.35, color=SURF, lw=0, zorder=1)         # no gridlines behind the event labels
    for t in STEPS:
        off = [str(j) for j, o in zip(COMP, d["off"][t]) if o]
        what = "" if not off else "\nall four off" if len(off) == 4 else "\n" + " and ".join(off) + " off"
        ax.text(t, 2.27, d["event"][t] + what, ha="center", va="top", linespacing=1.35, **note)
    # which thresholds are nearer to the new max than to the old one, one step after each new max
    after = STEPS[d["event"] == "new max"] + 1
    Mnew, Mold = M[after - 1], np.r_[0, M][after - 1]            # M_(t-1) and M_(t-2); 0 before the first input
    high = np.abs(d["theta"][after] - Mnew[:, None]) < np.abs(d["theta"][after] - Mold[:, None])
    assert (high == high[0]).all()
    names = lambda sel: ", ".join(sub("θ", j) for j in np.array(THR)[sel])
    ax.annotate(f"One step after a new max (steps {', '.join(map(str, after))}):\n{names(high[0])} are at the new max,\n"
                f"{names(~high[0])} at the old one.", xy=(after[1], d["theta"][after[1]].min()), xytext=(after[1] + 0.16, 0.52),
                va="top", linespacing=1.35, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=6),
                bbox=dict(fc=SURF, ec="none", pad=1.5), **note)
    oldest = (STEPS[1:] - d["p1"][:-1]).max()                    # largest age of the max at a step where the thresholds act
    ax.text(-0.42, -0.63, f"{sub('θ', 8)} stays near the runner-up (or zero): here the max is at most {oldest} steps old.\n"
            "For an older max it rises to the max and above (section 4.1).", va="top", linespacing=1.35,
            bbox=dict(fc=SURF, ec="none", pad=1.5), **note)
    ax.set_xlim(*xlim); ax.set_ylim(-1.25, 2.35); ax.set_xticks(STEPS)
    ax.set_ylabel("value, in input units")
    mk = dict(color=INK, ls="", ms=8, mec=SURF, mew=1.4)
    titled_legend(ax, [Line2D([], [], marker="D", label="new max", **{**mk, "ms": 7.5}),
                       Line2D([], [], marker="s", label="new runner-up", **mk),
                       Line2D([], [], marker="o", label="low: below the runner-up\n(or below zero)", **mk)],
                  "input xₜ (drawn just right of step t)", loc="upper left", bbox_to_anchor=(1.008, 0.58))
    hs = [Line2D([], [], color=THR_COLOR[j], lw=2, marker="" if j == 4 else "o", ms=7, mec=SURF, mew=1.4,
                 label=sub("θ", j) + (": unit 4 fires above it" if j == 4 else f": comparator {j}"))
          for j in np.array(THR)[np.argsort(-d["theta"][-1])]]
    hs += [Line2D([], [], ls="", marker="o", mfc=SURF, mec=MUTED, ms=6, mew=1.6,
                  label="threshold below the input:\nthat comparator is off")]
    titled_legend(ax, hs, "thresholds that xₜ meets", loc="upper left", bbox_to_anchor=(1.008, 0.36))

    # (b) pulses
    inner = outer[1].subgridspec(3, 1, hspace=0.2)
    for i, u in enumerate(PULSE):
        ax = fig.add_subplot(inner[i])
        style(ax, "b  Unit 4 fires on each new max, units 5 and 9 one step later; unit 5 also one step after a new runner-up" if i == 0 else "")
        h = d["H"][:, u]
        ax.bar(STEPS, h, width=0.2, color=INK2, zorder=3)
        for t in STEPS[h > 0]:
            ax.text(t, h[t] + 0.7, f"{h[t]:.2f}", ha="center", va="bottom", bbox=dict(fc=SURF, ec="none", pad=0.8), **note)
        ax.text(9.58, 7.5, f"unit {u}" + (f"\n= {w[4]:.3f} [xₜ − θ₄]₊" if u == 4 else ""), va="center", linespacing=1.35, **note)
        ax.set_xlim(*xlim); ax.set_ylim(0, 15.5); ax.set_xticks(STEPS); ax.set_yticks([0, 5, 10])
        ax.tick_params(labelbottom=i == 2)
        if i == 1:
            ax.set_ylabel("activation hₜ[u]")
    ax.set_xlabel("step t")

    # (c), (d) scores per input
    top_s, top_m = tops(d)
    for row, (key, top, title) in enumerate([
            ("s", top_s, "c  Scores sₜ = U hₜ, one row per input: from step 2 on, the largest entry is the second largest so far"),
            ("m", top_m, "d  Without the comparators (mₜ): among the earlier inputs, the largest entry is the largest of them")]):
        ax = fig.add_subplot(outer[2 + row])
        style(ax, title)
        ax.grid(False)
        C = d[key] - np.r_[d[key][0, 0], np.nanmean(older(d[key])[:, 1:], 0)]      # minus the mean over the earlier inputs
        mesh = ax.pcolormesh(np.arange(11) - 0.5, np.arange(11) - 0.5, np.ma.masked_invalid(C), cmap=DIV,
                             norm=Normalize(-VMAX, VMAX), edgecolors=SURF, linewidth=1.2)
        for t in STEPS:
            for p in range(t + 1):
                if abs(C[p, t]) > VMAX:                         # beyond the colour range: print the value
                    ax.text(t, p, minus(f"{C[p, t]:.0f}"), ha="center", va="center", fontsize=8.5, color="#ffffff")
            if key == "m":                                      # the newest input is not part of the comparison
                ax.add_patch(Rectangle((t - 0.5, t - 0.5), 1, 1, fill=False, hatch="////", ec=INK2, alpha=0.45, lw=0))
            if top[t] >= 0:
                ax.add_patch(Rectangle((t - 0.5, top[t] - 0.5), 1, 1, fill=False, ec=INK, lw=2, zorder=5))
            for p, letter in ((d["p1"][t], "M"), (d["p2"][t], "S")):
                if p >= 0:
                    ax.text(t, p, letter, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=6,
                            color="#ffffff" if abs(C[p, t]) > 0.6 * VMAX else INK)
        # key and notes, in the empty corner
        items = [("frame", "largest entry of the column" if key == "s" else "largest entry among the inputs before xₜ"),
                 ("S", "second largest input so far (true)"), ("M", "largest input so far (true)")]
        items += [("hatch", "the input arriving at this step: not compared")] if key == "m" else []
        for i, (kind, text) in enumerate(items):
            y = 5.65 + 0.82 * i
            if kind == "frame":
                ax.add_patch(Rectangle((-0.3, y - 0.27), 0.5, 0.54, fill=False, ec=INK, lw=2))
            elif kind == "hatch":
                ax.add_patch(Rectangle((-0.3, y - 0.27), 0.5, 0.54, fill=False, hatch="////", ec=INK2, alpha=0.45, lw=0))
            else:
                ax.text(-0.05, y, kind, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
            ax.text(0.38, y, text, va="center", **note)
        if key == "s":
            ax.text(-0.3, 8.2, "At each new max of this sequence all four comparators are off, so those columns are the same "
                    "here and in d.", va="center", **note)
        ax.text(-0.3, 9.02, f"Each input keeps its row: at step t its score is entry p + 9 − t of {key}ₜ; "
                "the diagonal is the input arriving at that step.", va="center", **note)
        ax.set_xlim(*xlim); ax.set_ylim(9.5, -0.5); ax.set_yticks(STEPS)
        ax.set_xticks(STEPS, [f"{t}\n{e}" for t, e in zip(STEPS, d["event"])])
        ax.set_ylabel("input: position p (grey: its value)", labelpad=34)
        for p in STEPS:
            ax.text(-0.04, p, minus(f"{X[p]:.1f}"), transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=8.5, color=MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        extend = {(0, 0): "neither", (1, 0): "min", (0, 1): "max", (1, 1): "both"}[
            int(np.nanmin(C) < -VMAX), int(np.nanmax(C) > VMAX)]
        cb = fig.colorbar(mesh, cax=ax.inset_axes([1.02, 0, 0.022, 1]), extend=extend)
        cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, length=0, labelsize=8.5)
        cb.set_label("score minus the mean score\nof the earlier inputs, in logits", color=INK2, fontsize=9)
    ax.set_xlabel("step t")
    fig.savefig(OUT / "worked_example_timeline.png", dpi=150)
    print("wrote", OUT / "worked_example_timeline.png")


def draw_clock(d, law):
    z, lap, ev, age = d["z"], d["lap"], d["event"], STEPS - d["p1"]
    fig, ax = plt.subplots(figsize=(12, 7.15))
    header(fig, d, "The clock in the worked example (section 5.5)", y=0.975, dy=0.036)
    fig.subplots_adjust(left=0.058, right=0.69, top=0.846, bottom=0.084)
    style(ax, "One step after a new max the clock restarts near 0°, then turns; a new runner-up shortens it and holds it back")
    ax.grid(False)
    xlim, ylim = (-0.2, 0.94), (-0.22, 0.6)

    # reference: circles of constant |z| and the rays of the phase law
    for r in (0.2, 0.4, 0.6, 0.8):
        ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=0.8))
        e = r * np.exp(-1j * np.radians(11))
        ax.text(e.real, e.imag, f"{r:.1f}", ha="center", va="center", fontsize=8, color=MUTED,
                bbox=dict(fc=SURF, ec="none", pad=1), zorder=2)
    for a, e in zip(range(1, 10), np.exp(1j * np.angle(law))):
        ax.plot([0, 2 * e.real], [0, 2 * e.imag], color=AXIS, lw=1, zorder=1)
        rx = (xlim[1] if e.real > 0 else xlim[0]) / e.real if abs(e.real) > 1e-9 else np.inf
        ry = (ylim[1] if e.imag > 0 else ylim[0]) / e.imag if abs(e.imag) > 1e-9 else np.inf
        r = min(rx, ry) - 0.05                                  # just inside the point where the ray leaves the view
        ax.text(r * e.real, r * e.imag, f"a = {a}", ha="center", va="center", fontsize=8.5, color=INK2,
                bbox=dict(fc=SURF, ec="none", pad=1.5), zorder=2)
    ax.plot([0], [0], "+", color=INK2, ms=9, mew=1.2, zorder=2)

    # the trajectory; each step takes the colour of the max that stands after it
    label = dict(textcoords="offset points", ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    for t in STEPS[1:]:
        ax.plot(z[t - 1:t + 1].real, z[t - 1:t + 1].imag, "-", color=RAMP[lap[t]], lw=2, zorder=3, solid_capstyle="round")
    for (t, g), off in zip(d["ghost"].items(), [(0, 11), (9, 7)]):
        ax.annotate("", xy=(z[t].real, z[t].imag), xytext=(g.real, g.imag), zorder=3,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1, shrinkA=6, shrinkB=7))
        ax.plot(g.real, g.imag, "o", mfc=SURF, mec=RAMP[lap[t]], ms=7.5, mew=1.8, zorder=4)
        ax.annotate(f"{t}′", (g.real, g.imag), xytext=off, **label)
    offs = [(-9, 9), (8, 9), (10, 4), (4, -15), (10, -4), (-11, -13), (-13, 4), (-13, -3), (4, -15), (-12, 3)]
    for t in STEPS:
        ax.plot(z[t].real, z[t].imag, EVENT_MARK[ev[t]], color=RAMP[lap[t]], ms=8.5 if ev[t] == "new max" else 9,
                mec=SURF, mew=1.6, zorder=5)
        ax.annotate(str(t), (z[t].real, z[t].imag), xytext=offs[t], **label)
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Re z / w₄"); ax.set_ylabel("Im z / w₄")

    # the plotted values as a table beside the plot; it is also the key to colours and marks
    cell = lambda x, y, s, **kw: ax.text(x, y, s, transform=ax.transAxes, va="center",
                                         **{**dict(fontsize=8.5, color=INK2, ha="right"), **kw})
    deg = lambda c: minus(f"{int(round(np.degrees(np.angle(c))))}°")
    cols, dy, y = (1.078, 1.245, 1.318, 1.385, 1.45), 0.0385, 0.975       # right edges of the number columns
    for x, s in zip(cols, ("step", "a", "|z| / w₄", "arg z", "φ(a)")):
        cell(x, y, s, color=INK)
    cell(1.097, y, "input", color=INK, ha="left")
    for t in STEPS:
        if ev[t] == "new max":                                   # group heading
            y -= 1.15 * dy
            ax.plot([1.02, 1.05], [y, y], color=RAMP[lap[t]], lw=2, transform=ax.transAxes, clip_on=False)
            cell(1.062, y, f"while the max so far is {sub('x', t)} = {X[t]:.1f}", color=INK, ha="left")
        rows = [(str(t), z[t], ev[t], dict(marker=EVENT_MARK[ev[t]], color=RAMP[lap[t]], mec=SURF))]
        if t in d["ghost"]:
            rows.append((f"{t}′", d["ghost"][t], f"if {sub('x', t)} were −1", dict(marker="o", mfc=SURF, mec=RAMP[lap[t]])))
        for name, zz, what, mark in rows:
            y -= dy
            ax.plot(1.035, y, transform=ax.transAxes, clip_on=False, ms=7.5, mew=1.6, **mark)
            cell(1.097, y, what, ha="left")
            for x, s in zip(cols, (name, str(age[t]), f"{abs(zz):.2f}", deg(zz) if abs(zz) > 0.01 else "–",
                                   deg(law[age[t] - 1]) if age[t] else "–")):
                cell(x, y, s)
    cell(1.02, y - 3.4 * dy, "a: steps since the max arrived.  φ(a): the phase law of section 4.1\n"
         "(the rays), the direction of z when the max is a steps old and the\ngap is large.  Circles: |z| / w₄.  "
         "Arrows: what the new runner-up\ndid to z at that step.  low: an input below the runner-up so far\n"
         "(or below zero).", ha="left", linespacing=1.45)
    fig.savefig(OUT / "worked_example_clock.png", dpi=150)
    print("wrote", OUT / "worked_example_clock.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF})
    OUT.mkdir(exist_ok=True)
    if "--redraw" in sys.argv and CACHE.exists():
        law = np.load(CACHE)["law"]
    else:
        law = phase_law()
        np.savez(CACHE, law=law)
    data = example()
    check(data)
    print_table(data, law)
    draw_timeline(data)
    draw_clock(data, law)
