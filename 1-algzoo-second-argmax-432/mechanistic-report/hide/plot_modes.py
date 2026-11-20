"""Three figures on the modes of the resting map A = W[o, o], o = the 13 non-pulse units (report sections 2, 3.2, 4.3).

plots/modes_spectrum.png   a: eigenvalues of A (dots, named) and of the shift matrix Sigma = U_o A pinv(U_o) (rings).
                           b: for each mode, the rms size of its piece of the final logits (U_o times the mode pattern;
                              norm over indices 0-8 with their mean removed) and the share of answers that change when
                              the piece is deleted from h_9.  400k sequences.
plots/mode_patterns.png    right eigenvectors r_m (top row) and dual vectors l_m (bottom row) of level, runner and clock
                           on the 13 non-pulse units, in the normalisation of modes.py (l_m . W[o, 4] = 1).
plots/gate_patterns.png    a: how often each unit is on, by age of the maximum (states with a positive maximum); last row: how
                              often the whole pattern is the resting one.
                           b: cumulative share of steps covered by the most common gate patterns.  2^20 sequences.  The numbers
                              of patterns needed for 50, 90 and 99% are written as a range ("74–75") where this sample cannot
                              tell them apart (cumulative share within two binomial standard errors).

Run:  python -B plot_modes.py            (recomputes, about 10 s, caches plot_modes.npz)
      python -B plot_modes.py --redraw   (only redraws from the cache)
Both print the values to compare with report.md.
"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D, ScaledTranslation, blended_transform_factory

from common import COMP, ORD, PULSE, WAVE, N, U, W, pattern_id, run, running_stats, sample, second_argmax, w
from fig_mechanism import AXIS, BLUE, INK, INK2, MUTED, ORANGE, SURF, legend, style
from modes import A, IS_COMPLEX, LAM, LEFT, NAMES, RIGHT, coords

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_modes.npz"

MODE = dict(level="level", runner="runner", clock="clock", h2="second harmonic", alt="alternating",
            h3="third harmonic", short="short clock", trans="transient", echo="echo")
Uo = U[:, ORD]
SIGMA = Uo @ A @ np.linalg.pinv(Uo)                    # shift matrix of section 3.2
RESTING = sum(1 << u for u in ORD)                     # gate pattern: the 13 non-pulse units on, 4, 5, 9 off
TOL = 0.06                                             # eigenvalues of A and of Sigma closer than this count as the same
OMEGA = np.degrees(np.angle(LAM["clock"]))
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
MK = dict(ms=8, mec=SURF, mew=1.6)
minus = lambda s: s.replace("-", "−")


# ---------------------------------------------------------------- data
def compute():
    out = {}
    # pieces of the final logits: logits = sum over modes of U_o (zeta_m r_m) + the pulse units' own part
    R = [Uo @ RIGHT[m] for m in NAMES]
    size2 = lambda v: ((v[:, :9] - v[:, :9].mean(1, keepdims=True)) ** 2).sum()    # indices 0-8, mean removed
    ss, ch, z2, tot, acc, n = np.zeros(10), np.zeros(10), np.zeros(9), 0.0, 0, 0
    for seed in range(7101, 7105):
        x = sample(100_000, seed)
        h9 = run(x)[:, -1]
        lg, zeta = h9 @ U.T, coords(h9)
        C = [(2 if IS_COMPLEX[m] else 1) * np.real(np.outer(zeta[m], r)) for m, r in zip(NAMES, R)]
        C.append(h9[:, PULSE] @ U[:, PULSE].T)
        assert np.abs(sum(C) - lg).max() < 1e-8         # the ten pieces add up to the logits
        ans = lg.argmax(1)
        ss += [size2(c) for c in C]
        ch += [((lg - c).argmax(1) != ans).sum() for c in C]
        z2 += [(np.abs(zeta[m]) ** 2).sum() for m in NAMES]
        tot += size2(lg); acc += (ans == second_argmax(x)).sum(); n += len(x)
    out.update(piece_rms=np.sqrt(ss / n), changed=ch / n, logit_rms=np.sqrt(tot / n), zeta_rms=np.sqrt(z2 / n),
               acc=acc / n)

    # gates by age of the max, and the census of gate patterns
    on, n_age, rest_age, rest_S, n_S, rest_t, comp, pids = (np.zeros((10, N)), np.zeros(10), np.zeros(10),
                                                            np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(2), [])
    for seed in range(7201, 7205):
        x = sample(2 ** 18, seed)
        G = run(x) > 0
        pid, st = pattern_id(G), running_stats(x)
        rest = pid == RESTING
        for a in range(10):
            s = (st["a1"] == a) & (st["M1"] > 0)                    # states with a positive maximum
            on[a] += G[s].sum(0); n_age[a] += s.sum(); rest_age[a] += rest[s].sum()
            s = (st["a1"] == a) & (st["M2"] > 0.05)                 # verifier V5's condition, for the check only
            rest_S[a] += rest[s].sum(); n_S[a] += s.sum()
        rest_t += rest.sum(0)
        comp += [G[..., COMP].all(-1).sum(), (~G[..., COMP]).all(-1).sum()]
        pids.append(pid.astype(np.int32))
    pid = np.concatenate(pids)
    ids, cnt = np.unique(pid, return_counts=True)
    o = np.argsort(-cnt, kind="stable")
    out.update(on=on / n_age[:, None], age_share=n_age / n_age.sum(), rest_age=rest_age / n_age,
               rest_age_S=rest_S / np.maximum(n_S, 1), rest_t=rest_t / 2 ** 20, comp=comp / pid.size,
               pat_id=ids[o], pat_share=cnt[o] / pid.size, n_steps=pid.size)
    np.savez(CACHE, **out)
    return out


# ---------------------------------------------------------------- drawing helpers
def titled(ax, title, sub="", lines=None):
    """Bold finding as the title (style of fig_mechanism) with secondary lines under it; `lines` reserves their height."""
    style(ax, title)
    n = lines if lines is not None else sub.count("\n") + 1 if sub else 0
    pad = 11 + 12.5 * n if n else 8
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=pad)
    ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, pad - 5), textcoords="offset points",
                fontsize=8.5, color=INK2, va="top", linespacing=1.25)


def dial(ax, x, y, phi, col, R=8.5):
    """Phase dial at data x, axes-fraction y: a circle of radius R points with an arrow at angle phi (0 right, 90 deg up)."""
    tr = (Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans
          + ScaledTranslation(x, y, blended_transform_factory(ax.transData, ax.transAxes)))
    c, s = np.cos(phi), np.sin(phi)
    arrow = R * np.array([[-.6, .08], [.3, .08], [.3, .34], [.95, 0], [.3, -.34], [.3, -.08], [-.6, -.08]])
    ax.add_artist(Circle((0, 0), R, transform=tr, fc=SURF, ec=AXIS, lw=0.9, clip_on=False, zorder=4))
    ax.add_artist(Polygon(arrow @ np.array([[c, s], [-s, c]]), transform=tr, fc=col, ec="none", clip_on=False, zorder=5))


def rbar(ax, pos, v, thick, color, horizontal=False, r_px=4):
    """Bar from 0 to v at `pos`: square at the baseline, corners rounded by r_px pixels at the data end.
    Call it after the axis limits are set."""
    bb = ax.get_window_extent()
    rx, ry = r_px * np.ptp(ax.get_xlim()) / bb.width, r_px * np.ptp(ax.get_ylim()) / bb.height
    ra, rc = (rx, ry) if horizontal else (ry, rx)                    # radius along and across the bar
    ra, rc, lo, hi = min(ra, abs(v)) * np.sign(v), min(rc, thick / 2), pos - thick / 2, pos + thick / 2
    V = np.array([(0, lo), (v - ra, lo), (v, lo), (v, lo + rc), (v, hi - rc), (v, hi), (v - ra, hi), (0, hi), (0, lo)])
    ax.add_patch(PathPatch(MPath(V if horizontal else V[:, ::-1], [1, 2, 3, 3, 2, 3, 3, 2, 79]), fc=color, ec="none", zorder=3))


def header(fig, title, sub):
    """Figure title and a secondary line, top left."""
    FW, FH = fig.get_size_inches()
    fig.text(0.35 / FW, 1 - 0.3 / FH, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.35 / FW, 1 - 0.62 / FH, sub, fontsize=9, color=INK2, va="top", linespacing=1.4)


def boxes(fig):
    """Axes placed by a rectangle in inches: box(left, bottom, width, height)."""
    FW, FH = fig.get_size_inches()
    return lambda l, b, wd, h: fig.add_axes([l / FW, b / FH, wd / FW, h / FH])


# ---------------------------------------------------------------- figure 1
def fig_spectrum(d):
    fig = plt.figure(figsize=(16.5, 7.95))
    box = boxes(fig)
    header(fig, "The modes of the resting map: their eigenvalues, and their pieces of the final logits",
           "Resting map A = W[o, o]: the one-step map of the 13 non-pulse units o of M₁₆,₁₀ when all of them are on and units 4, 5, 9 "
           "are off (report sections 2, 3.2 and 4.3).  N(0, 1) inputs.")

    # (a) the two spectra
    xl, yl, H0 = (-1.3, 1.78), (-1.25, 1.25), 5.55
    ax = box(0.62, 0.8, H0 * (xl[1] - xl[0]) / (yl[1] - yl[0]), H0)
    lamA, lamS = np.linalg.eigvals(A), np.linalg.eigvals(SIGMA)
    near = lambda l, others: np.abs(others - l).min() < TOL
    only_A = [MODE[m] for m in NAMES if not near(LAM[m], lamS)]
    only_S = sorted(l.real for l in lamS if l.imag >= 0 and not near(l, lamA))
    titled(ax, f"a  Spectrum of the resting map A; its shift matrix Σ has no {', '.join(only_A[:-1])} or {only_A[-1]}",
           "dots: the 13 eigenvalues of A, with the mode names of section 2;  rings: the 10 eigenvalues of Σ = Uₒ A Uₒ⁺ (section 3.2)\n"
           f"a ring on a dot: Σ has an eigenvalue within {TOL} of it;  labels give |λ| ∠ rotation per step, conjugates are not labelled")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=1.2, zorder=2)
    ax.axhline(0, color=AXIS, lw=1, zorder=2); ax.axvline(0, color=AXIS, lw=1, zorder=2)
    ax.plot(lamA.real, lamA.imag, "o", color=BLUE, zorder=4, label="resting map A", **MK)
    ax.plot(lamS.real, lamS.imag, "o", ms=13.5, mfc="none", mec=ORANGE, mew=1.8, zorder=3, label="shift matrix Σ")
    place = dict(level=(.065, .16, "left"), runner=(.1015, -.16, "left"), clock=(.09, .085, "left"),
                 h2=(0, .185, "center"), alt=(0, .115, "left"), h3=(0, -.125, "center"), short=(-.1, .045, "right"),
                 trans=(.015, .115, "center"), echo=(-.04, .115, "right"))        # label offsets from the eigenvalue
    for m, (dx, dy, ha) in place.items():
        l = LAM[m]
        if m in ("level", "runner"):                                               # the two dots overlap: leader lines
            ax.plot([l.real, l.real + dx - 0.007], [0, dy - 0.035 * np.sign(dy)], color=INK2, lw=0.8, zorder=3.5)
        val = (f"{abs(l):.3f} ∠ {np.degrees(np.angle(l)):.2f}°" if IS_COMPLEX[m]
               else minus(f"{l.real:.4f}" if l.real > 0.9 else f"{l.real:.3f}"))
        ax.annotate(MODE[m], (l.real + dx, l.imag + dy), ha=ha, va="bottom", fontsize=9.5, color=INK)
        ax.annotate(val, (l.real + dx, l.imag + dy), xytext=(0, -2), textcoords="offset points", ha=ha, va="top",
                    fontsize=8.5, color=INK2)
    for l in only_S:
        ax.annotate(f"{l:.2f}", (l, 0.06), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.annotate("only in Σ", (np.mean(only_S), 0.135), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_aspect("equal"); ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xticks(np.arange(-1, 1.6, 0.5)); ax.set_yticks(np.arange(-1, 1.1, 0.5))
    ax.set_xlabel("Re λ"); ax.set_ylabel("Im λ")
    legend(ax, loc="lower right", handletextpad=0.5, borderaxespad=0.8, labelspacing=0.9)

    # (b) each mode's piece of the final logits
    names = [MODE[m] for m in NAMES] + ["pulse units\n4, 5, 9"]
    y, cols = np.r_[np.arange(9), 9.75], [BLUE] * 9 + [MUTED]
    assert d["piece_rms"][NAMES.index("short")] > d["logit_rms"]            # what the title says
    panels = [(d["piece_rms"], lambda v: f"{v:.1f}", 104, "rms size of the piece (logits)",
               f"b  The modes' pieces of the final logits cancel: the short clock's alone ({d['piece_rms'][NAMES.index('short')]:.0f}) "
               f"exceeds all logits ({d['logit_rms']:.0f})",
               "size: norm of the piece over indices 0–8, mean over\nindices removed; rms over 400k sequences"),
              (100 * d["changed"], lambda v: f"{v:.1f}%", 60, "answers changed (% of sequences)", "",
               "share of answers that change when the piece\nis deleted from h₉")]
    for k, (v, fmt, xmax, xlab, title, sub) in enumerate(panels):
        ax = box(9.45 + 3.6 * k, 0.8, 3.25, H0)
        titled(ax, title, sub)
        ax.grid(False, axis="y")
        ax.set_xlim(0, xmax); ax.set_ylim(10.45, -1.15)
        for yi, vi, col in zip(y, v, cols):
            rbar(ax, yi, vi, 0.46, col, horizontal=True)
            ax.annotate(fmt(vi), (vi, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2,
                        bbox=dict(fc=SURF, ec="none", pad=1.5), zorder=5)
        ax.set_yticks(y); ax.set_yticklabels(names if k == 0 else [], fontsize=9.5, color=INK)
        ax.set_xlabel(xlab)
        if k == 0:
            ax.axvline(d["logit_rms"], color=INK2, lw=1, zorder=2)
            ax.annotate(f"all logits: {d['logit_rms']:.1f}", (d["logit_rms"], -0.72), xytext=(-5, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "modes_spectrum.png", dpi=150)
    print("wrote", OUT / "modes_spectrum.png")


# ---------------------------------------------------------------- figure 2
def fig_patterns(d):
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 9.75))
    fig.subplots_adjust(left=0.047, right=0.99, top=0.828, bottom=0.164, wspace=0.17, hspace=0.6)
    header(fig, "Level, runner and clock: their patterns r and dual vectors ℓ on the 13 non-pulse units",
           "The three slow modes of the resting map A = W[o, o] of M₁₆,₁₀ (report sections 2 and 4.3):  state h[o] = Σ ζ r (real modes) "
           "+ 2 Re(z r) (pairs),  coordinate ζ = ℓ·h[o].")
    units = COMP + WAVE
    idx = [ORD.index(u) for u in units]
    # what the titles of a, b, d, e and f say about the vectors
    rl, rr, ll, lr, lc = (v[idx] for v in (RIGHT["level"].real, RIGHT["runner"].real, LEFT["level"].real, LEFT["runner"].real,
                                           np.abs(LEFT["clock"])))
    top = lambda v, n: {units[k] for k in np.argsort(-np.abs(v))[:n]}
    assert (rl > 0).all() and (rr[:4] > 0).all() and (rr[4:] < 0).sum() == 6
    assert top(ll, 1) == {2} and top(lr, 3) == {2, 7, 8} and lc[:4].min() > 2 * lc[4:].max()
    xs, cols = np.r_[np.arange(4), 4.8 + np.arange(9)], [BLUE] * 4 + [ORANGE] * 9
    zr = dict(zip(NAMES, d["zeta_rms"] / w[4]))
    on_data = lambda m, z="ζ": f"on data {z} has rms {zr[m]:.2f} w₄ (final states)"
    text = {(0, "level"): ("a  Level pattern r: positive on all 13 units", on_data("level")),
            (0, "runner"): ("b  Runner pattern r: six wave units against the rest", on_data("runner")),
            (0, "clock"): ("c  Clock pattern r: spread over all 13 units",
                           f"bars: amplitude 2|r|;  arrows: the phase of z at which the unit peaks (degrees)\n{on_data('clock', 'z')}"),
            (1, "level"): ("d  Level dual vector ℓ: sits on unit 2", "ζ = Σ ℓ[u] h[u]"),
            (1, "runner"): ("e  Runner dual vector ℓ: unit 2 and comparators 7, 8", "ζ = Σ ℓ[u] h[u]"),
            (1, "clock"): ("f  Clock dual vector ℓ: mostly the four comparators",
                           "bars: |ℓ|;  arrows: the direction in which the unit's activation moves z (degrees)\nz = Σ ℓ[u] h[u]")}
    for i, vec in enumerate((RIGHT, LEFT)):
        for j, m in enumerate(("level", "runner", "clock")):
            ax, v = axs[i, j], vec[m][idx]
            titled(ax, *text[i, m], lines=2)
            ax.grid(False, axis="x")
            ax.set_xlim(-0.75, xs[-1] + 0.75)
            tb = blended_transform_factory(ax.transData, ax.transAxes)
            if IS_COMPLEX[m]:
                v, ph = (2, 1)[i] * np.abs(v), (-1, 1)[i] * np.angle(v)          # peak phase -arg r; push direction arg l
                ax.set_ylim(0, 1.52 * v.max())
                ax.set_yticks([t for t in ax.get_yticks() if t <= 1.1 * v.max()])    # no gridlines behind the dials
                for x, a, p in zip(xs, v, ph):
                    dial(ax, x, 0.9, p, INK if a >= 0.25 * v.max() else MUTED)
                    ax.text(x, 0.79, minus(f"{np.degrees(p):.0f}"), transform=tb, ha="center", va="center", fontsize=7.5, color=INK2)
            else:
                v = v.real
                ax.axhline(0, color=AXIS, lw=1, zorder=2)
                lo, hi = min(v.min(), 0), max(v.max(), 0)
                ax.set_ylim(lo - 0.16 * (hi - lo) * (lo < 0), hi + 0.16 * (hi - lo))
                for k in np.argsort(-np.abs(v))[:3]:                      # label the three largest entries only
                    ax.annotate(minus(f"{v[k]:.2f}"), (xs[k], v[k]), xytext=(0, 3 if v[k] > 0 else -3), textcoords="offset points",
                                ha="center", va="bottom" if v[k] > 0 else "top", fontsize=8.5, color=INK2)
            for x, a, col in zip(xs, v, cols):
                rbar(ax, x, a, 0.6, col)
            ax.set_xticks(xs); ax.set_xticklabels(units, fontsize=9.5, color=INK)
            for xm, lab, col in ((0.75, "comparators", BLUE), (8.2, "wave units", ORANGE)):
                ax.plot([xm - 0.32], [-0.135], "s", color=col, ms=7, transform=tb, clip_on=False)
                ax.text(xm, -0.135, lab, transform=tb, ha="left", va="center", fontsize=9, color=INK2)
            c = "|z|" if IS_COMPLEX[m] else "ζ"
            ax.set_ylabel((f"activation per unit of {c}", f"{c} per unit of activation")[i])
    fig.text(0.35 / 16.5, 0.05, "Normalisation of hide/modes.py: ℓ·W[o, 4] = 1 for every mode, so a pulse of unit 4 adds its size to every coordinate, "
             "and the nine patterns (r for real modes, 2 Re r for pairs) add up to W[o, 4].\n"
             f"Arrows in c: the unit is largest when arg z = −arg r;  arg z = 0° one step after a pulse of unit 4 and advances {OMEGA:.2f}° per "
             "resting step.  Arrows in f: arg ℓ.  Grey arrows: amplitude below a quarter of the panel's largest.\n"
             "The entries of r and ℓ depend on how each unit's activation is scaled; the products ℓ[u] r[u], which sum to 1 over the "
             "13 units, do not.", fontsize=8.5, color=INK2, va="center", linespacing=1.45)
    fig.savefig(OUT / "mode_patterns.png", dpi=150)
    print("wrote", OUT / "mode_patterns.png")


# ---------------------------------------------------------------- figure 3
def describe(p):
    """A gate pattern as its departures from the resting one."""
    on, off = [u for u in PULSE if p >> u & 1], [u for u in COMP + WAVE if not p >> u & 1]
    parts = ["on: " + ", ".join(map(str, on))] * bool(on) + ["off: " + ", ".join(map(str, off))] * bool(off)
    return "  ·  ".join(parts) or "the resting pattern"


def covers(d):
    """Patterns needed to cover 50, 90 and 99% of all steps: the count in this sample, and the same as text, written as a
    range ("74–75") when shifting the cumulative share by two binomial standard errors either way changes the count."""
    cum = np.cumsum(d["pat_share"])
    need = lambda f: int(np.searchsorted(cum, f) + 1)
    se2 = lambda f: 2 * np.sqrt(f * (1 - f) / d["n_steps"])
    cover = {f: need(f) for f in (0.5, 0.9, 0.99)}
    return cover, {f: "–".join(map(str, sorted({need(f - se2(f)), need(f + se2(f))}))) for f in cover}


def fig_gates(d):
    fig = plt.figure(figsize=(16.5, 8.9))
    box = boxes(fig)
    header(fig, "Gate patterns: which units are on at which age of the maximum, and how many patterns occur",
           "M₁₆,₁₀ on 2²⁰ sequences of N(0, 1) inputs.  A unit is on when its activation is above zero.  Resting pattern: the 13 non-pulse "
           "units on and units 4, 5, 9 off; its linear map A = W[o, o] defines the modes (report section 4.3).")

    # (a) how often each unit is on, by age of the max
    ax = box(1.3, 0.9, 6.9, 6.35)
    ra = d["rest_age"]                                    # share of the resting pattern by age of the max
    none, rare = np.flatnonzero(ra < 0.001), np.flatnonzero((ra >= 0.001) & (ra < 0.05))
    span = lambda a: f"{a[0]}–{a[-1]}" if len(a) > 1 and np.all(np.diff(a) == 1) else ", ".join(map(str, a))
    titled(ax, f"a  The gates follow the age of the maximum: no resting pattern at ages {span(none)}, and rarely at ages {span(rare)}",
           "share of states in which the unit is on, among states with a positive maximum so far;  last row: share of them in the resting pattern\n"
           "numbers (%): cells where the unit departs from the resting pattern in more than 5% of states, and the whole last row")
    ax.grid(False)
    P, y0, yt, xe = d["on"].T, 0.0, [], np.arange(11) - 0.5
    rows = [("comparators", COMP), ("pulse units", PULSE), ("wave units", WAVE), ("", [-1])]   # -1: the resting pattern
    for g, (name, us) in enumerate(rows):
        data = d["rest_age"][None] if us == [-1] else P[us]
        mesh = ax.pcolormesh(xe, y0 - 0.5 + np.arange(len(us) + 1), data, cmap=SEQ, vmin=0, vmax=1, edgecolors=SURF, linewidth=1.6)
        for r, u in enumerate(us):
            for a in range(10):
                p = data[r, a]
                if u == -1 or (p > 0.05 if u in PULSE else p < 0.95):
                    ax.text(a, y0 + r, "0" if p == 0 else "<1" if p < 0.005 else f"{100 * p:.0f}", ha="center", va="center",
                            fontsize=8, color="#ffffff" if p > 0.66 else INK)   # 0.66: where white and ink contrast equally
        yt += [(y0 + r, "resting\npattern" if u == -1 else str(u)) for r, u in enumerate(us)]
        ax.annotate(name, (-0.5, y0 + (len(us) - 1) / 2), xytext=(-34, 0), textcoords="offset points", rotation=90,
                    ha="center", va="center", fontsize=9, color=INK2)
        y0 += len(us) + (0.45, 0.45, 0.8, 0)[g]
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(y0 - 0.5, -0.5)
    ax.set_yticks([t[0] for t in yt]); ax.set_yticklabels([t[1] for t in yt], fontsize=9.5, color=INK)
    ax.set_xticks(range(10)); ax.set_xticklabels([f"{a}\n{100 * s:.0f}%" for a, s in enumerate(d["age_share"])])
    ax.set_xlabel("steps since the maximum arrived  (second line: share of these states)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(mesh, cax=box(8.32, 4.05, 0.13, 3.2), ticks=[0, 0.25, 0.5, 0.75, 1], format=FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    cb.ax.set_title("unit on", fontsize=8.5, color=INK2, pad=6, loc="left")

    # (b) coverage by the most common patterns
    ax = box(9.85, 0.9, 6.25, 6.35)
    share, ids = d["pat_share"], d["pat_id"]
    cum, rank = np.cumsum(share), np.arange(1, len(share) + 1)
    cover, count = covers(d)
    titled(ax, f"b  {count[0.5]} gate patterns cover half of all steps, {count[0.9]} cover 90%",
           f"{len(share)} different patterns occur in the 2²⁰ × 10 steps;  cumulative share of steps, most common pattern first\n"
           "a count given as a range is uncertain by sampling: ± 2 binomial standard errors of the cumulative share")
    ax.plot(rank, 100 * cum, color=BLUE, lw=2, zorder=3)
    k = int(np.flatnonzero(ids == RESTING)[0])
    note = dict(textcoords="offset points", va="center", fontsize=8.5, color=INK2, bbox=dict(fc=SURF, ec="none", pad=1.5))
    ax.plot(rank[k], 100 * cum[k], "o", color=BLUE, zorder=4, **MK)
    ax.annotate(f"resting pattern:\n{100 * share[k]:.1f}% of steps", (rank[k], 100 * cum[k]), xytext=(10, -3), **note)
    for f, n in cover.items():
        ax.plot(n, 100 * cum[n - 1], "o", color=BLUE, zorder=4, **MK)
        ax.annotate(f"{count[f]} patterns: {100 * f:.0f}%", (n, 100 * cum[n - 1]), xytext=(-10, 5) if f == 0.5 else (9, -7),
                    ha="right" if f == 0.5 else "left", **note)
    ax.set_xscale("log"); ax.set_xlim(0.8, 2600); ax.set_ylim(0, 104)
    ax.tick_params(which="minor", color=AXIS, length=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.set_yticks(range(0, 101, 25))
    ax.set_xlabel("number of gate patterns, most common first"); ax.set_ylabel("cumulative share of all steps (%)")
    ax.add_patch(Rectangle((0.37, 0.004), 0.63, 0.468, transform=ax.transAxes, fc=SURF, ec="none", zorder=2.5))  # hides the grid
    ta = dict(transform=ax.transAxes, fontsize=8.5, va="center", zorder=5)
    ax.text(0.385, 0.44, f"the {cover[0.5]} patterns that cover half of all steps,\nas departures from the resting pattern",
            color=INK, linespacing=1.3, **ta)
    for r in range(cover[0.5]):
        yy = 0.385 - 0.039 * r
        ax.text(0.41, yy, str(r + 1), ha="right", color=INK2, **ta)
        ax.text(0.43, yy, describe(int(ids[r])), color=INK2, **ta)
        ax.text(0.985, yy, f"{100 * share[r]:.1f}%", ha="right", color=INK2, **ta)
    fig.savefig(OUT / "gate_patterns.png", dpi=150)
    print("wrote", OUT / "gate_patterns.png")


# ---------------------------------------------------------------- values to compare with report.md
def checks(d):
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    pol = lambda c: f"{abs(c):.3f}@{np.degrees(np.angle(c)):.2f}"
    print("\naccuracy on the 400k sequences:", round(float(d["acc"]), 4))
    print("eigenvalues of A:    ", {MODE[m]: (pol(LAM[m]) if IS_COMPLEX[m] else round(float(LAM[m].real), 4)) for m in NAMES})
    lamS = np.linalg.eigvals(SIGMA)
    print("eigenvalues of Sigma:", [pol(l) if abs(l.imag) > 1e-9 else round(float(l.real), 3)
                                    for l in sorted(lamS, key=lambda l: -abs(l)) if l.imag >= 0])
    print(f"distance from each eigenvalue of A to the nearest one of Sigma (same mode if below {TOL}):",
          {MODE[m]: round(float(np.abs(lamS - LAM[m]).min()), 3) for m in NAMES})
    print("piece of the final logits: rms size, answers changed (%)   [report 4.3]")
    for nm, r, c in zip([MODE[m] for m in NAMES] + ["pulse units"], d["piece_rms"], d["changed"]):
        print(f"   {nm:16s} {r:6.1f} {100 * c:6.1f}")
    print(f"   all logits       {d['logit_rms']:6.1f}")
    print("rms |zeta| / w4 of the final states:", dict(zip(NAMES, (d["zeta_rms"] / w[4]).round(3).tolist())))
    print("level r relative to its entry on unit 2, units", ORD, ":", (RIGHT["level"].real / RIGHT["level"][ORD.index(2)].real).round(2))
    for m in ("level", "runner", "clock"):
        c = [LEFT[m][ORD.index(k)] * -w[k] / w[4] for k in COMP]
        print(f"   {m}: l[k] g_k / w4 for comparators 1, 6, 7, 8 [table of section 5]:",
              [pol(v) if IS_COMPLEX[m] else round(float(v.real), 3) for v in c])
    r = RIGHT["clock"]
    print("clock: |U_o r| / sum_u |U[:, u] r_u| =", round(float(np.linalg.norm(Uo @ r) / np.linalg.norm(Uo * r, axis=0).sum()), 4))
    tot = sum((2 if IS_COMPLEX[m] else 1) * RIGHT[m].real for m in NAMES)
    print("sum of the nine patterns = W[o, 4]:", bool(np.allclose(tot, W[ORD, 4])))
    print("resting pattern: share of all states", round(float(d["pat_share"][d["pat_id"] == RESTING][0]), 4), " by t:", d["rest_t"].round(3))
    print("   by age of the max, M > 0:   ", d["rest_age"].round(4))
    print("   by age of the max, S > 0.05:", d["rest_age_S"].round(4))
    print("all four comparators on / off (share of all steps):", d["comp"].round(3))
    cum, (cover, count) = np.cumsum(d["pat_share"]), covers(d)
    print("distinct patterns:", len(cum), " needed for 50 / 90 / 99 %:", list(cover.values()), " within two standard errors:", list(count.values()))
    print("   cumulative share of the n - 1, n, n + 1 most common, for these n:", [cum[n - 2:n + 1].round(5).tolist() for n in cover.values()])
    on = d["on"]
    print("P(on | age of the max, M > 0): unit 4 at age 0", on[0, 4].round(4), "| units 5, 9 at age 1", on[1, [5, 9]].round(4),
          "| unit 0 at age 2", on[2, 0].round(4), "| unit 11 at age 5", on[5, 11].round(4), "| unit 10 at age 6", on[6, 10].round(4))


if __name__ == "__main__":
    data = dict(np.load(CACHE)) if ("--redraw" in sys.argv and CACHE.exists()) else compute()
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF})
    fig_spectrum(data); fig_patterns(data); fig_gates(data)
    checks(data)
