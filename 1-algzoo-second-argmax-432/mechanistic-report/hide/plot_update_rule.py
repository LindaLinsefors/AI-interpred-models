"""Three figures on the exact update rule of the mode coordinates (report section 5).

    zeta_m(t) = lam_m zeta_m(t-1) + (l_m.w[o]) x_t + sum_{u in 4,5,9} (l_m.W[o,u]) h_{t-1}[u]
                + sum_{k in 1,6,7,8} l_m[k] g_k [x_t - theta_k(t)]_+ + sum_{u in wave} l_m[u] [-a_t[u]]_+

Coordinates and pulses are in units of w_4 = 10.157, as in the report's coefficient table. Every number drawn is
computed here from the weights, from the impulse x = (1, 0, ..., 0), or from sampled N(0,1) sequences.

  ../plots/update_rule_clock.png         the clock row in the complex plane: (a) one arrow per term and the free
                                         rotation as an arc from 1 to lam, nothing rescaled; (b) zoom on the terms too
                                         small to see in (a); (c) the four comparator arrows tip to tail; (d) the exact
                                         terms of step 2 of the impulse, i.e. the first step after a new maximum
  ../plots/update_rule_level_runner.png  the level and runner rows as signed horizontal bars
  ../plots/update_rule_check.png         the rule checked on 200k sequences (2M one-step updates): prediction from
                                         h_{t-1} and x_t against the actual coordinate, with all five terms (blue) and
                                         with the first three only (grey)

Run:  python -B plot_update_rule.py            (about 10 s; caches the binned check in plot_update_rule.npz)
      python -B plot_update_rule.py --redraw   (redraws from the cache; coefficients always come from the weights)
"""
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, Patch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

from common import *
from modes import LAM, LEFT
from fig_mechanism import style, legend, SURF, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE, AQUA, YELLOW, RAMP, DIV

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_update_rule.npz"
MODES = ["level", "runner", "clock"]
W4, G = w[4], -w[COMP]
CI, WI = [ORD.index(k) for k in COMP], [ORD.index(u) for u in WAVE]
NB, N_SEQ, SEED = 90, 200_000, 20261120
GREYS = LinearSegmentedColormap.from_list("greys", ["#d6d5cd", "#6f6e69"])
BLUES = LinearSegmentedColormap.from_list("blues", ["#9ec5f4", "#2a78d6", "#0d366b"])
C_FREE, C_IN, C_PULSE, C_COMP, C_CLIP = INK2, AQUA, BLUE, ORANGE, YELLOW       # colour = kind of term, in all figures
MASK = dict(facecolor=SURF, edgecolor="none", pad=1.5)                         # behind a label: hides the lines it crosses
RULE = (r"$ = \lambda\,{z}_{t-1} + c_x\,x_t + \sum_{u \in \{4,5,9\}} c_u\,h_{t-1}[u] + \sum_{k \in \{1,6,7,8\}} "
        r"c_k\,[x_t - \theta_k]_+ + \sum_{u \in \mathrm{wave}} (\ell[u]/w_4)\,[-a_t[u]]_+$")
RULE_ZETA = RULE.replace("{z}", r"\zeta")
COEF = (r"$c_x = \ell^\top w[o]\,/\,w_4, \qquad c_u = \ell^\top W[o,u], \qquad c_k = \ell[k]\,g_k\,/\,w_4, "
        r"\qquad g_k = -w_k$")


def coefficients(n):
    """Coefficient-table column of mode n: free evolution, input, pulses of 4, 5, 9, comparators 1, 6, 7, 8 off."""
    L = LEFT[n]
    return dict(lam=LAM[n], inp=L @ w[ORD] / W4, pulse=L @ W[np.ix_(ORD, PULSE)], comp=L[CI] * G / W4)


def terms(x):
    """The five terms at every step, from h_{t-1} and x_t only, and the actual coordinates; all in units of w_4.
    Returns pred[mode] (5, B, T), act[mode] (B, T), off (B, T, 4) = [x_t - theta_k]_+, clip (B, T, 9) = [-a_t[u]]_+."""
    H = run(x)
    Hp = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)             # h_{t-1}, with h_{-1} = 0
    a = x[..., None] * w + Hp @ W.T                                          # pre-activations a_t
    off = np.maximum(x[..., None] - Hp @ W[COMP].T / G, 0)                   # theta_k(t) = (W h_{t-1})_k / g_k
    clip = np.maximum(-a[..., WAVE], 0)
    pred, act = {}, {}
    for n in MODES:
        L, c = LEFT[n], coefficients(n)
        pred[n] = np.stack([c["lam"] * (Hp[..., ORD] @ L) / W4, c["inp"] * x, Hp[..., PULSE] @ c["pulse"] / W4,
                            off @ c["comp"], clip @ L[WI] / W4])
        act[n] = H[..., ORD] @ L / W4
    return pred, act, off, clip


def compute():
    """Check on data: 2-D histograms of prediction against actual, largest errors, how often the gate terms act."""
    pred, act, off, clip = terms(sample(N_SEQ, SEED))
    out = dict(hist=[], lim=[], err5=[], err3=[])
    for n, part in (("level", np.real), ("runner", np.real), ("clock", np.real), ("clock", np.imag)):
        y, p5, p3 = (part(v).ravel() for v in (act[n], pred[n].sum(0), pred[n][:3].sum(0)))
        lo, hi = min(y.min(), p3.min()), max(y.max(), p3.max())
        lim = [lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)]
        out["hist"].append([np.histogram2d(y, p, NB, [lim, lim])[0] for p in (p3, p5)])
        out["lim"].append(lim); out["err5"].append(np.abs(p5 - y).max()); out["err3"].append(np.abs(p3 - y).max())
    out["off_diagonal"] = sum(h[1].sum() - np.trace(h[1]) for h in out["hist"])       # five-term points off the diagonal
    out["err5_modulus"] = [np.abs(pred[n].sum(0) - act[n]).max() for n in MODES]
    out["rms_terms"] = [np.sqrt((np.abs(pred[n]) ** 2).mean((1, 2))) for n in MODES]
    n_off = (off > 0).sum(-1)
    out["comp_all_on"], out["comp_all_off"] = (n_off == 0).mean(), (n_off == 4).mean()
    out["no_gate_term"] = ((n_off == 0) & (clip == 0).all(-1)).mean()
    out = {k: np.array(v) for k, v in out.items()}
    np.savez_compressed(CACHE, **out)
    return out


# ---------------------------------------------------------------- drawing helpers
def polar(c, nd=2, na=0):
    return f"{abs(c):.{nd}f} ∠ {np.degrees(np.angle(c)):.{na}f}°".replace("-", "−")


def signed(v):
    return f"{v:+.{3 if abs(v) < 1 else 2}f}".replace("-", "−")


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m} × 10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def arrow(ax, tip, color, tail=0j, lw=2.0, ms=13, zorder=4, ring=False):
    """Arrow between two complex numbers; ring=True lays a wider copy in the surface colour under it (overlapping arrows)."""
    ends = (tail.real, tail.imag), (tip.real, tip.imag)
    kw = dict(arrowstyle="-|>", mutation_scale=ms, shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt")
    if ring:
        ax.add_patch(FancyArrowPatch(*ends, color=SURF, lw=lw + 2, zorder=zorder - 0.1, **kw))
    ax.add_patch(FancyArrowPatch(*ends, color=color, lw=lw, zorder=zorder, **kw))


def spiral(ax, start, color, lw=2.0, ms=13):
    """One step of free rotation, drawn as the path start * lam^s, s = 0..1, with an arrowhead at start * lam."""
    p = start * LAM["clock"] ** np.linspace(0, 1, 60)
    ax.plot(p.real[:-4], p.imag[:-4], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    arrow(ax, p[-1], color, tail=p[-5], lw=lw, ms=ms, zorder=3)


def note(ax, at, text, dx, dy, ha="left", va="center", **kw):
    """Text at the complex number `at`, shifted by (dx, dy) points."""
    kw = dict(dict(fontsize=8.5, color=INK2), **kw)
    return ax.annotate(text, (at.real, at.imag), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **kw)


def plane(ax, title, lim, tick):
    style(ax, title)
    ax.set_aspect("equal")
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.xaxis.set_major_locator(plt.MultipleLocator(tick)); ax.yaxis.set_major_locator(plt.MultipleLocator(tick))
    ax.axhline(0, color=AXIS, lw=1); ax.axvline(0, color=AXIS, lw=1)


def hbar(ax, y, v, h, color, r_pt=1.9):
    """Horizontal bar from 0 to v: square at the baseline, corners rounded by r_pt points (4 px) at the data end."""
    bb, r = ax.get_window_extent(), r_pt * ax.figure.dpi / 72
    rx = min(r * np.ptp(ax.get_xlim()) / bb.width, abs(v)) * np.sign(v)
    ry = min(r * np.ptp(ax.get_ylim()) / bb.height, h / 2)
    V = [(0, y - h / 2), (v - rx, y - h / 2), (v, y - h / 2), (v, y - h / 2 + ry), (v, y + h / 2 - ry), (v, y + h / 2),
         (v - rx, y + h / 2), (0, y + h / 2), (0, y - h / 2)]
    ax.add_patch(PathPatch(MPath(V, [1, 2, 3, 3, 2, 3, 3, 2, 79]), fc=color, ec="none", zorder=3))


def header(fig, title, rule, gap, notes=()):
    """Bold figure title, the update rule under it, and further lines given as (x, y, text)."""
    fig.text(0.012, 0.975, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.975 - gap, rule, fontsize=9.5, color=INK2, va="top")
    for x, y, s in notes:
        fig.text(x, y, s, fontsize=9.5 if "$" in s else 9, color=INK2, va="top")


# ---------------------------------------------------------------- figure 1
def fig_clock():
    c = coefficients("clock")
    lam, cin, (c4, c5, c9), cc = c["lam"], c["inp"], c["pulse"], c["comp"]
    csum, deg = cc.sum(), lambda v: np.degrees(np.angle(v))
    imp = np.zeros((1, T)); imp[0, 0] = 1
    pred, act, _, clip = terms(imp)
    zi, ti, hi = act["clock"][0], pred["clock"][:, 0], run(imp)[0]    # impulse: z(t), its five terms, the states

    fig = plt.figure(figsize=(14.6, 8.3))
    header(fig, "Update rule, clock row: what each term adds to the clock coordinate z in one step", "$z_t$" + RULE, 0.045,
           [(0.555, 0.972, COEF + r"$, \qquad \ell = \ell_{\mathrm{clock}}$"),
            (0.555, 0.93, f"z = ℓ·h[o] and the pulses h[u] are in units of w₄ = {W4:.3f}.    r ∠ φ means r·exp(iφ)."
                          "    No arrow is rescaled.")])
    axa = fig.add_axes([0.045, 0.07, 0.445, 0.783])
    axb, axc, axd = (fig.add_axes([x, y, 0.195, 0.343]) for x, y in ((0.555, 0.50), (0.79, 0.50), (0.555, 0.07)))

    # (a) every coefficient as an arrow from the origin, true lengths
    ax = axa
    plane(ax, "a  Unit 4 writes 1; the four comparator terms are as large and point four ways", (-1.9, 1.9, -1.9, 1.9), 0.5)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=1)
    note(ax, np.exp(2.2j), "|c| = 1", -3, 3, ha="right", va="bottom")
    for a in range(1, T):                                             # dial: direction of z at age a of the maximum
        e = zi[a] / abs(zi[a])
        ax.plot([1.60 * e.real, 1.66 * e.real], [1.60 * e.imag, 1.66 * e.imag], color=MUTED, lw=1.2)
        ax.text(1.75 * e.real, 1.75 * e.imag, str(a), ha="center", va="center", fontsize=8.5, color=INK2, bbox=MASK)
    for v in cc:
        arrow(ax, v, C_COMP)
    arrow(ax, c9, C_PULSE); arrow(ax, c4, C_PULSE); arrow(ax, c5, C_PULSE, ms=6)
    spiral(ax, 1 + 0j, C_FREE)
    ax.add_patch(Rectangle((-0.095, -0.095), 0.19, 0.19, fill=False, ec=INK2, lw=0.8, zorder=5))
    note(ax, -0.095 + 0.095j, "b", -2, 2, ha="right", va="bottom", fontweight="bold")
    note(ax, c4, "pulse of unit 4\n1 (by definition)", 7, 5, va="bottom", bbox=MASK)      # labels hide circle and grid
    note(ax, c9, f"pulse of unit 9\n{polar(c9)}", 9, -7, va="top", bbox=MASK)
    note(ax, cc[0], f"comparator 1 off\n{polar(cc[0])}", 0, -7, ha="center", va="top", bbox=MASK)
    note(ax, cc[1], f"comparator 6 off\n{polar(cc[1])}", 6, 8, ha="center", va="bottom", bbox=MASK)
    note(ax, cc[2], f"comparator 7 off\n{polar(cc[2])}", 8, 3, bbox=MASK)
    note(ax, cc[3], f"comparator 8 off\n{polar(cc[3])}", 7, -5, va="top", bbox=MASK)
    note(ax, lam, f"free rotation, one step\nλ = {polar(lam, 3, 2)}", 6, 13, ha="center", va="bottom", bbox=MASK)
    note(ax, c5, f"pulse of unit 5\n{polar(c5, 3)}", -22, -3, ha="right", va="top", bbox=MASK)
    ax.set_xlabel("Re c    (change of z / w₄ per unit of the driving quantity)"); ax.set_ylabel("Im c")
    ax.text(0.5, 0.012, "outer numbers 1–9: direction of z when the maximum is that many steps old (impulse response)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK2, bbox=MASK)

    # (b) zoom on the origin
    ax = axb
    plane(ax, "b  Input and unit 5 add almost nothing", (-0.095, 0.095, -0.095, 0.095), 0.05)
    arrow(ax, c5, C_PULSE); arrow(ax, csum, C_COMP, lw=1.6, ms=8); arrow(ax, cin, C_IN, lw=1.6, ms=8, zorder=5, ring=True)
    note(ax, c5, f"pulse of unit 5\n{polar(c5, 3)}", 6, -9, ha="center", va="top")
    lead = dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=1, shrinkB=3)
    note(ax, csum, f"sum of the four\ncomparator arrows\n{polar(csum, 3)}", 14, 36, va="bottom",
         arrowprops=dict(lead, relpos=(0, 0)))
    note(ax, cin, f"input, per unit of xₜ\n{polar(cin, 3)}", -1, -36, va="top", arrowprops=dict(lead, relpos=(0.03, 1)))
    zoom = (axb.get_position().width / 0.19) / (axa.get_position().width / 3.8)
    ax.text(0.03, 0.985, f"box b of panel a, enlarged ×{zoom:.0f}", transform=ax.transAxes, va="top", fontsize=8.5,
            color=INK2, bbox=MASK)

    # (c) the four comparator arrows tip to tail
    ax = axc
    plane(ax, "c  The four comparator arrows cancel", (-0.75, 1.5, -0.55, 1.7), 0.5)
    order = [2, 1, 0, 3]                                              # comparators 7, 6, 1, 8: by angle
    p = np.concatenate([[0], np.cumsum(cc[order])])
    for i, j in enumerate(order):
        arrow(ax, p[i + 1], C_COMP, tail=p[i])
        mid, out = (p[i] + p[i + 1]) / 2, -1j * cc[j] / abs(cc[j])    # label on the outside of the loop
        note(ax, mid, str(COMP[j]), 11 * out.real, 11 * out.imag, ha="center")
    ax.plot([0], [0], "+", color=INK2, ms=9, mew=1.2)
    ax.text(0.5, 0.032, f"placed tip to tail they leave a gap of {abs(csum):.3f}:\n"   # the report's condition for cancelling
            "all four off by the same xₜ − θₖ hardly move z", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=INK2, bbox=MASK)
    ax.text(0.5, 0.6, "numbers:\ncomparator\nswitched off", ha="center", va="center", fontsize=8.5, color=INK2)

    # (d) step 2 of the impulse response, term by term
    ax = axd
    plane(ax, f"d  After a new maximum, unit 9's pulse holds the first step back to {deg(zi[2] / zi[1]):.1f}°",
          (0.555, 0.875, 0.42, 0.74), 0.1)
    clips = clip[0, 2] * LEFT["clock"][WI] / W4                       # the clip term, unit by unit
    steps = [ti[0, 2], c5 * hi[1, 5] / W4, c9 * hi[1, 9] / W4, clips.sum()]
    p = np.cumsum(steps)
    assert abs(p[-1] - zi[2]) < 1e-12 and abs(ti[1, 2]) + abs(ti[3, 2]) < 1e-12      # nothing else acts at this step
    for q, r in ((p[0], 0.775), (p[-1], 0.728)):
        ax.plot([0, q.real], [0, q.imag], color=MUTED, lw=1)
        ax.text(r * q.real / abs(q), r * q.imag / abs(q), f"{deg(q):.1f}°", rotation=deg(q), rotation_mode="anchor",
                ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.text(0.97, 0.965, "grey lines: from the origin,\nat the phases of λ z₁ and z₂", transform=ax.transAxes, ha="right",
            va="top", fontsize=8.5, color=INK2)
    spiral(ax, zi[1], C_FREE)
    arrow(ax, p[1], C_PULSE, tail=p[0], ms=11); arrow(ax, p[2], C_PULSE, tail=p[1], ms=11)
    arrow(ax, p[3], C_CLIP, tail=p[2], ms=9)
    ax.plot([p[-1].real], [p[-1].imag], "o", color=INK, ms=6, mec=SURF, mew=1.2, zorder=6)
    note(ax, p[0], "λ z₁", 2, 8, ha="center", va="bottom")
    note(ax, (p[0] + p[1]) / 2, "unit 5", -8, 5, ha="right", va="bottom")
    note(ax, (p[1] + p[2]) / 2, "unit 9", 0, -7, ha="center", va="top")
    note(ax, (p[2] + p[3]) / 2, "clipped\nwave units", 9, -8, va="top")
    note(ax, p[-1], "z₂", 8, 6, va="bottom")
    ax.set_xlabel("Re z / w₄"); ax.set_ylabel("Im z / w₄")

    # key: the terms of panel d as numbers, and the colours
    rows = [("Impulse x = (1, 0, …, 0), step 2, exact:", ""), ("free rotation, λ z₁", polar(p[0], 3, 1)),
            ("+ pulse of unit 5", polar(steps[1], 3, 1)), ("+ pulse of unit 9", polar(steps[2], 3, 1))]
    rows += [(f"+ wave unit {u} clipped at zero", polar(v, 3, 1)) for u, v in zip(WAVE, clips) if v != 0]
    rows += [("= z₂", polar(zi[2], 3, 1)), ("z₂ / z₁", polar(zi[2] / zi[1], 3, 1))]
    for i, (a, b) in enumerate(rows):
        y = 0.392 - 0.0285 * i - 0.01 * (i == len(rows) - 1)
        fig.text(0.79, y, a, fontsize=9, color=INK if i == 0 else INK2, va="top", fontweight="bold" if i == 0 else None)
        fig.text(0.985, y, b, fontsize=9, color=INK2, va="top", ha="right")
    kinds = {"pulse of unit 4, 5 or 9: c per unit of h[u] / w₄": C_PULSE,
             "comparator switched off: c per unit of xₜ − θₖ": C_COMP,
             "input: c per unit of xₜ (visible in b only)": C_IN,
             "wave units clipped at zero (d only)": C_CLIP, "free rotation, z → λ z": C_FREE}
    fig.legend([plt.Line2D([], [], color=col, lw=2) for col in kinds.values()], list(kinds), loc="lower left",
               bbox_to_anchor=(0.782, 0.05), frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.6, labelspacing=0.4)
    fig.savefig(OUT / "update_rule_clock.png", dpi=150)
    plt.close(fig)
    many = lambda vs: " ".join(polar(v, 4, 1) for v in vs)
    print("clock   lam", polar(lam, 4, 2), "| input", polar(cin, 4, 1), "| pulses 4, 5, 9:", many(c["pulse"]),
          "\n        comparators 1, 6, 7, 8 off:", many(cc), "| sum", polar(csum, 4, 1),
          "\n        impulse z(1..9):", many(zi[1:]), "\n        step 2 (free, unit 5, unit 9, clips):", many(steps),
          "| clips by unit:", {u: polar(v, 4, 1) for u, v in zip(WAVE, clips) if v != 0}, "| z2/z1", polar(zi[2] / zi[1], 4, 2))


# ---------------------------------------------------------------- figure 2
def fig_bars():
    rows = (["free evolution: per unit of ζ(t−1)", "input: per unit of xₜ"]
            + [f"pulse of unit {u}: per unit of h[{u}] / w₄" for u in PULSE]
            + [f"comparator {k} off: per unit of xₜ − θ{s}" for k, s in zip(COMP, "₁₆₇₈")]
            + ["sum of the four comparator terms"])
    ypos = np.array([0, 1.3, 2.6, 3.6, 4.6, 5.9, 6.9, 7.9, 8.9, 10.3])
    cols = [C_FREE, C_IN] + [C_PULSE] * 3 + [C_COMP] * 5
    fig = plt.figure(figsize=(13.0, 5.9))
    header(fig, "Update rule, level and runner rows: the coefficient of each term", r"$\zeta_t$" + RULE_ZETA, 0.062,
           [(0.555, 0.972, COEF), (0.555, 0.912, f"ζ = ℓ·h[o] and the pulses h[u] are in units of w₄ = {W4:.3f}.    "
                                                 "Note the two horizontal scales.")])
    for i, (n, lim, tick) in enumerate((("level", (-0.34, 1.27), 0.25), ("runner", (-7.0, 5.6), 2))):
        c = coefficients(n)
        v = np.real(np.r_[c["lam"], c["inp"], c["pulse"], c["comp"], c["comp"].sum()])
        title = f"a  Level: unit 4 writes 1; no other term writes more than {abs(np.delete(v, [0, 2])).max():.2f}"
        if n == "runner":
            assert set(np.argsort(-abs(v[1:9]))[:2]) == {6, 7}                 # comparators 7 and 8 are the largest
            title = "b  Runner: comparators 7 and 8 write most; the four sum to " + f"{v[9]:+.2f}".replace("-", "−")
        ax = fig.add_axes([0.215 + 0.395 * i, 0.105, 0.375, 0.67])
        style(ax, title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(*lim); ax.set_ylim(ypos[-1] + 0.75, -0.75)
        ax.xaxis.set_major_locator(plt.MultipleLocator(tick))
        ax.axvline(0, color=AXIS, lw=1)
        ax.plot(lim, [9.6, 9.6], color=AXIS, lw=1)                    # rule above the sum, as in a ledger
        for j, (y, val, col) in enumerate(zip(ypos, v, cols)):
            hbar(ax, y, val, 0.44, col)
            text = f"{val:.4f}" if j == 0 else "1 (by definition)" if j == 2 else signed(val)
            ax.annotate(text, (val, y), xytext=(6 * (1 if val > 0 else -1), 0), textcoords="offset points", va="center",
                        ha="left" if val > 0 else "right", fontsize=8.5, color=INK2, bbox=MASK)
        ax.set_yticks(ypos, rows if i == 0 else [""] * len(rows))
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"coefficient in the update of the {n} coordinate ζ / w₄")
        print(f"{n:7s} lam {v[0]:.4f} | input {v[1]:+.4f} | pulses 4, 5, 9:", " ".join(f"{q:+.4f}" for q in v[2:5]),
              "| comparators 1, 6, 7, 8 off:", " ".join(f"{q:+.4f}" for q in v[5:9]), f"| sum {v[9]:+.4f}")
    hs = [Patch(fc=col, ec="none") for col in (C_FREE, C_IN, C_PULSE, C_COMP)]
    fig.legend(hs, ["free evolution", "input", "pulse of unit 4, 5 or 9", "comparator switched off"], loc="upper left",
               bbox_to_anchor=(0.548, 0.875), ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.1,
               columnspacing=1.6)
    fig.savefig(OUT / "update_rule_level_runner.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_check(d):
    names = ["Level", "Runner", "Clock, real part", "Clock, imaginary part"]
    sym = ["level coordinate ζ(t) / w₄", "runner coordinate ζ(t) / w₄", "Re z(t) / w₄", "Im z(t) / w₄"]
    fig = plt.figure(figsize=(9.6, 10.9))
    text = ["Blue: all five terms, computed from h(t−1) and xₜ, against the coordinate after the step.  "
            "Grey: the first three terms only (free evolution,",
            f"input, pulses); these are exact on the {100 * d['no_gate_term']:.0f}% of updates in which no comparator is "
            "switched off and no wave unit is clipped.",
            f"{N_SEQ:,} N(0,1) sequences × {T} steps.  Coordinates in units of w₄ = {W4:.3f}.  Darker = more updates "
            f"per bin (log scale, 1 to {sci(d['hist'].max())})."]
    header(fig, f"The update rule is exact: {N_SEQ * T:,} one-step updates on data", r"$\zeta_t$" + RULE_ZETA, 0.034,
           [(0.012, 0.906, COEF + r"$; \qquad \ell,\ \lambda$ of the coordinate shown")]
           + [(0.012, 0.878 - 0.019 * j, s) for j, s in enumerate(text)])
    norm = LogNorm(1, d["hist"].max())
    for i in range(4):
        ax = fig.add_axes([0.085 + 0.5 * (i % 2), 0.45 - 0.41 * (i // 2), 0.39, 0.3435])
        style(ax, f"{'abcd'[i]}  {names[i]}: largest error {sci(d['err5'][i])}")
        lo, hi = d["lim"][i]
        for layer, cmap in enumerate((GREYS, BLUES)):                 # grey: three terms; blue, on top: all five
            ax.imshow(np.ma.masked_equal(d["hist"][i, layer].T, 0), origin="lower", extent=[lo, hi, lo, hi], cmap=cmap,
                      norm=norm, interpolation="nearest", zorder=2 + layer)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, zorder=2.5)   # identity
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"actual {sym[i]}"); ax.set_ylabel(f"predicted {sym[i]}")
        hs = [Patch(fc=BLUES(0.6), ec="none"), Patch(fc=GREYS(0.5), ec="none")]
        legend(ax, handles=hs, labels=["all five terms", f"first three terms only:\nlargest error {d['err3'][i]:.2f}"],
               loc="upper left", borderaxespad=0.3)
    fig.savefig(OUT / "update_rule_check.png", dpi=150)
    plt.close(fig)
    print("check   largest error, five terms (level, runner, Re z, Im z):", " ".join(f"{v:.2e}" for v in d["err5"]),
          f"| complex clock: {d['err5_modulus'][2]:.2e} | five-term points off the diagonal bins: {int(d['off_diagonal'])}",
          "\n        largest error, three terms:", " ".join(f"{v:.3f}" for v in d["err3"]),
          "\n        rms of the five terms, rows level, runner, clock:\n", np.round(d["rms_terms"], 4),
          f"\n        all four comparators on: {d['comp_all_on']:.4f} | all four off: {d['comp_all_off']:.4f}"
          f" | no gate term: {d['no_gate_term']:.4f}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF})
    OUT.mkdir(exist_ok=True)
    data = dict(np.load(CACHE)) if ("--redraw" in sys.argv and CACHE.exists()) else compute()
    fig_clock(); fig_bars(); fig_check(data)
    print("wrote update_rule_clock.png, update_rule_level_runner.png, update_rule_check.png in", OUT)
