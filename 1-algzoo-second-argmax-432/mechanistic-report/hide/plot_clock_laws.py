"""Three figures for the measured laws of section 4.1 of the report (written to ../plots/).

clock_laws.png         The clock coordinate z = zeta_clock on random inputs, with the law
                           z ~ w4 [alpha(a) e^{i phi(a)} (M - S+) + beta S+],   a = steps since the maximum arrived.
                       (a) phi(a), the phase of the gap term, against a, with the line 42.75 deg * (a - 1) (the angle
                           of the clock eigenvalue). Inset: phi(a) minus that line; bars = range over the runner-up's age.
                       (b) alpha(a): mean over the runner-up's ages, and separately for the cells in which no runner-up
                           arrived after the maximum, with the decay of the eigenvalue (0.951 per step);
                           c(a): the coefficient of the one-term law z ~ M - c S+.
                       (c) median of phi(a) - arg z (the lag of z) against the relative gap (M - S+)/M, for maxima that
                           are 1, 2-4 and 5-9 steps old.
                       (d) median |z| / (w4 M) against S+/M for ages 1, 4, 8, with the line 1 - S+/M: what is left at a tie.
thresholds_by_age.png  Positions f = (theta - S)/(M - S) of the thresholds of units 1, 6, 7, 8 and 4 by age of the
                       maximum: medians and quartiles over states with S > 0 and M - S > 0.2 (0 = runner-up,
                       1 = maximum). Shaded: inputs below the maximum that are above unit 4's threshold (false firing).
level_and_unit2.png    (a) 2-d histogram of the level coordinate / w4 against max(M_(t-1), 0), with the fitted line and the
                           median ratio one step after a new maximum (all states; the legend also gives it for the states
                           with a runner-up above 0.05, the selection behind the report's 0.99);
                       (b, c) the same for h_t[2] / 17.3, for maxima that are 1-4 and 5-9 steps old;
                       (d) median h_t[2] / (17.3 M_(t-1)) against S+/M of the earlier inputs, by age of the maximum:
                           the sag of unit 2 is towards the runner-up.

Definitions follow e05_cell_traces.py and e03_clock_radius.py (and the verifiers' v5_b, v5_c, v8_extra): the fits
use the states with S > 0.05, the binned medians the states with M > 0.3. Every plotted number is computed here
(17.3 is the fitted slope of unit 2); report() prints them next to the report's.

Run:  python -B plot_clock_laws.py            (1M sequences, about half a minute; caches plot_clock_laws.npz)
      python -B plot_clock_laws.py --redraw   (only redraws from the cache)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle

from common import *
from modes import coords, LAM
from fig_mechanism import style, legend, SURF, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE, AQUA, YELLOW, RAMP, DIV

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_clock_laws.npz"

W4 = w[4]                                                     # 10.157: coordinate per unit of excess over theta_4
OMEGA, RHO = np.degrees(np.angle(LAM["clock"])), abs(LAM["clock"])   # 42.75 deg, 0.951
SEEDS, NB = (4101, 4102, 4103, 4104), 250_000                 # 1M sequences in four chunks
AGES = np.arange(1, 10)
UNITS = COMP + [4]                                            # thresholds of units 1, 6, 7, 8 and 4
CLASSES = [(1, 1), (2, 4), (5, 9)]                            # age classes of the maximum (panel c of figure 1)
RAD_AGES = (1, 4, 8)                                          # one age per class (panel d of figure 1)
GAP_EDGES = np.linspace(0, 1, 11)
RAD_EDGES = np.r_[np.arange(0, 0.95, 0.1), 0.95]              # bins of S/M: <= 0, (0, .1], .., (.8, .9], (.9, .95], (.95, 1]
RAD_X = np.r_[0, np.arange(0.05, 0.9, 0.1), 0.925, 0.975]
SAG_EDGES = np.arange(0, 0.95, 0.1)                           # bins of S/M: <= 0, (0, .1], .., (.9, 1]
SAG_X = np.r_[0, np.arange(0.05, 1, 0.1)]
HMAX, HBINS = 3.5, 140                                        # 2-d histograms: 0..3.5 in steps of 0.025


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def binned(v, key, edges, stat=np.median):
    """stat of v in the bins of key: key <= edges[0], (edges[0], edges[1]], ..., > edges[-1]."""
    b = np.digitize(key, edges, right=True)
    return np.array([stat(v[b == k]) for k in range(len(edges) + 1)])


def compute():
    acc = {}
    for seed in SEEDS:                                        # chunks keep the memory low
        x = sample(NB, seed)
        H = run(x)
        st, c = running_stats(x), coords(H)
        new = dict(M=st["M1"], S=st["M2"], a1=st["a1"].astype(np.int8), a2=st["a2"].astype(np.int8),
                   z=c["clock"] / W4, lev=c["level"] / W4, h2=H[:, :, 2],
                   th=np.stack([-(H[:, :-1] @ W[j]) / w[j] for j in UNITS], -1))    # theta_j(t), t = 1..9
        for k, v in new.items():
            acc.setdefault(k, []).append(v)
    M, S, a1, a2, z, lev, h2, th = (np.concatenate(acc[k]) for k in ("M", "S", "a1", "a2", "z", "lev", "h2", "th"))
    Sp = np.maximum(S, 0)                                     # S = -inf at t = 0
    out = {}

    # ---- figure 1: z / w4 ~ alpha (M - S) + beta S, complex least squares on states with S > 0.05
    ok = S > 0.05

    def alpha(s):
        X = np.stack([M[s] - S[s], S[s]], 1)
        return np.linalg.solve(X.T @ X, X.T @ z[s])[0]

    pooled = np.array([alpha(ok & (a1 == a)) for a in AGES])  # pooled over the runner-up's age a2
    cells = np.full((9, 10), np.nan + 0j)                     # one fit per cell (a, a2) with at least 2000 states
    for i, a in enumerate(AGES):
        for b in range(10):
            s = ok & (a1 == a) & (a2 == b)
            if b != a and s.sum() >= 2000:
                cells[i, b] = alpha(s)
    phi = np.angle(pooled)
    dev = np.degrees(np.angle(cells * np.exp(-1j * phi)[:, None]))
    out["phi"] = np.degrees(np.unwrap(phi))
    out["phi_lo"], out["phi_hi"] = np.nanmin(dev, 1), np.nanmax(dev, 1)           # range over a2, relative to phi(a)
    out["alpha"] = np.nanmean(np.abs(cells), 1)
    out["alpha_late"] = np.array([np.nanmean(np.abs(cells[i, a + 1:])) for i, a in enumerate(AGES[:8])])   # a2 > a
    cg = np.arange(0, 1.2001, 0.001)                          # one-term law z ~ alpha' (M - c S): best real c
    cbest = []
    for a in AGES:
        s = ok & (a1 == a)
        num = np.abs((z[s] * M[s]).sum() - cg * (z[s] * S[s]).sum()) ** 2
        den = (M[s] ** 2).sum() - 2 * cg * (M[s] * S[s]).sum() + cg ** 2 * (S[s] ** 2).sum()
        cbest.append(cg[(num / den).argmax()])                # largest uncentred R2
    out["c"] = np.array(cbest)

    big = (M > 0.3) & (a1 >= 1)
    lag = np.degrees(wrap(phi[np.clip(a1, 1, 9) - 1] - np.angle(z)))              # > 0: z is behind phi(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap, sm = (M - Sp) / M, S / M
    out["lag"] = np.array([binned(lag[s], gap[s], GAP_EDGES[1:-1])
                           for s in (big & (a1 >= lo) & (a1 <= hi) for lo, hi in CLASSES)])
    out["absdev"] = np.array([binned(np.abs(lag[s]), gap[s], [0.1, 0.3, 0.6]) for s in (big & (a1 == a) for a in AGES)])
    out["rad"] = np.array([binned((np.abs(z) / M)[s], sm[s], RAD_EDGES) for s in (big & (a1 == a) for a in AGES)])

    # ---- figure 2: thresholds between runner-up (0) and maximum (1); theta(t) acts on x_t, the stored values are those of t - 1
    Mo, So, ao = M[:, :-1], S[:, :-1], a1[:, :-1] + 1         # ao: age of the maximum at the step the threshold acts
    okf = (So > 0) & (Mo - So > 0.2)
    with np.errstate(invalid="ignore"):
        f = (th - So[..., None]) / (Mo - So)[..., None]
    out["f"] = np.array([[np.percentile(f[..., k][okf & (ao == a)], [25, 50, 75]) for a in AGES] for k in range(5)])
    out["th4_M"] = np.array([np.median((th[..., 4] / Mo)[(Mo > 0.3) & (ao == a)]) for a in AGES])
    # the report's table: S = 0.5, M = 1.5, all other inputs -1, every placement and every step at which both are stored
    pM, pS = np.nonzero(~np.eye(10, dtype=bool))
    xc = -np.ones((90, 10))
    xc[np.arange(90), pM], xc[np.arange(90), pS] = 1.5, 0.5
    Hc = run(xc)
    fc = np.stack([-(Hc[:, :-1] @ W[j]) / w[j] for j in UNITS], -1) - 0.5         # M - S = 1
    tt = np.arange(1, 10)[None, :]
    stored, agec = tt > np.maximum(pM, pS)[:, None], tt - pM[:, None]
    out["f_ctrl"] = np.array([[fc[..., k][stored & (agec == a)].mean() for a in AGES] for k in range(5)])
    out["n_ctrl"] = stored.sum()

    # ---- figure 3: level coordinate and unit 2 against the maximum of the EARLIER inputs, states t >= 1
    X, Xs, age = np.maximum(Mo, 0), np.maximum(So, 0), ao     # age of that maximum at step t
    lv, u2 = lev[:, 1:], h2[:, 1:]
    for name, y in (("lev", lv), ("u2", u2)):
        k = (y * X).sum() / (X * X).sum()                     # least squares through the origin
        out[name + "_slope"], out[name + "_r2"] = k, 1 - ((y - k * X) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    k2 = out["u2_slope"]                                      # the report's 17.3
    hist = lambda y, s: np.histogram2d(X[s], y[s], bins=HBINS, range=[[0, HMAX]] * 2)[0]
    out["hist_lev"], out["hist_young"], out["hist_old"] = hist(lv, age >= 1), hist(u2 / k2, age <= 4), hist(u2 / k2, age >= 5)
    pos = Mo > 0.3
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio, levr, smo = u2 / (k2 * X), lv / X, So / Mo
    out["sag"] = np.array([binned(ratio[s], smo[s], SAG_EDGES) for s in (pos & (age == a) for a in AGES)])
    out["u2_med"] = np.array([np.median(ratio[pos & (age == a)]) for a in AGES])
    out["lev_med"] = np.array([np.median(levr[pos & (age == a)]) for a in AGES])
    # the report's "0.99 w4 M one step after a new maximum" is for the states with a runner-up above 0.05 (verifier V5):
    # level_t / (w4 M_t) over the states with a1(t) = 1 and S_t > 0.05
    out["lev_med_s"] = np.median((lv / M[:, 1:])[(a1[:, 1:] == 1) & (S[:, 1:] > 0.05)])
    out["u2_ab"] = np.array([np.linalg.lstsq(np.stack([X[s], Xs[s]], 1), u2[s], rcond=None)[0]
                             for s in (pos & (age == a) for a in AGES)])            # h_t[2] ~ a M + b S+
    np.savez(CACHE, **out)
    return out


# ---------------------------------------------------------------- drawing
MK = dict(ms=7.5, mec=SURF, mew=1.6)
COLS = [BLUE, ORANGE, AQUA, YELLOW, INK2]                     # units 1, 6, 7, 8 as in mechanism_summary.png; unit 4 grey
OLD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal ramp for ages 5..9
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])
FALSE = "#d03b3b"


def setup():
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF,
                         "mathtext.fontset": "custom", "mathtext.rm": "Segoe UI", "mathtext.it": "Segoe UI:italic",
                         "mathtext.bf": "Segoe UI:bold"})


def header(fig, x, title, sub):
    fig.text(x, 0.975, title, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(x, 0.975 - 0.33 / fig.get_figheight(), sub, fontsize=9.5, color=INK2, ha="left", va="top")


BOX = dict(fc=SURF, ec="none", pad=1.5)                       # behind text that would otherwise sit on grid lines


def note(ax, x, y, s, **kw):
    return ax.text(x, y, s, fontsize=8.5, color=INK2, **kw)


def leg(ax, title=None, **kw):
    """fig_mechanism's legend on a patch of the surface colour, so that no grid line runs through its text."""
    lg = legend(ax, title=title, **kw)
    lg.set_frame_on(True)
    lg.get_frame().set(facecolor=SURF, edgecolor="none", alpha=1)
    lg.get_title().set_color(INK2); lg.get_title().set_fontsize(8.5)
    return lg


def draw_clock(d):
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 10.6))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.875, bottom=0.06, wspace=0.22, hspace=0.3)
    header(fig, 0.065, "Measured laws of the clock coordinate z (section 4.1)",
           f"z ≈ w₄ [ α(a) · exp(iφ(a)) · (M − S⁺) + β S⁺ ],   a = steps since the maximum arrived,   M and S = largest and "
           f"second-largest input so far,   S⁺ = max(S, 0),   w₄ = {W4:.3f}\n"
           f"{len(SEEDS) * NB // 10 ** 6}M random sequences, inputs N(0, 1).   a, b: complex least squares on the states with S > 0.05.   "
           "c, d: medians over the states with M > 0.3.")
    step1, later = d["phi"][1] - d["phi"][0], np.polyfit(AGES[1:], d["phi"][1:], 1)[0]

    ax = axs[0, 0]
    style(ax, f"a  Phase of the gap term: first step {step1:.0f}°, then {later:.0f}° per step (eigenvalue: {OMEGA:.2f}°)")
    ax.plot([1, 9.35], [0, OMEGA * 8.35], "-", color=AXIS, lw=2, label=f"{OMEGA:.2f}° · (a − 1): the angle of the clock eigenvalue")
    ax.plot(AGES, d["phi"], "o", color=BLUE, label="φ(a), from the fit on states with S > 0.05", **MK)
    ax.set_xlim(0.6, 9.6); ax.set_ylim(-18, 425); ax.set_xticks(AGES); ax.set_yticks(range(0, 361, 60))
    ax.set_xlabel("a, steps since the maximum arrived"); ax.set_ylabel("φ(a), phase of the gap term (degrees)")
    leg(ax, loc="upper left")
    ins = ax.inset_axes([0.585, 0.10, 0.385, 0.25])              # placed so that its tick labels miss the main grid lines
    style(ins, "")
    res = d["phi"] - OMEGA * (AGES - 1)
    ins.axhline(0, color=AXIS, lw=2)
    ins.vlines(AGES, res + d["phi_lo"], res + d["phi_hi"], color=BLUE, lw=1.6)
    ins.plot(AGES, res, "o", color=BLUE, ms=6, mec=SURF, mew=1.3)
    ins.set_xlim(0.4, 9.6); ins.set_ylim(-13, 5); ins.set_xticks(AGES); ins.set_yticks([-10, -5, 0])
    ins.tick_params(labelsize=7.5)
    ins.set_title(f"φ(a) − {OMEGA:.2f}° · (a − 1), degrees\nbars: range over the runner-up's age", loc="left", fontsize=8.5, color=INK2, pad=5,
                  bbox=BOX)
    plt.setp(ins.get_xticklabels() + ins.get_yticklabels(), bbox=BOX)     # the main panel's grid runs behind the inset's labels

    ax = axs[0, 1]
    style(ax, "b  Both coefficients of the radius shrink with the age of the maximum")
    ax.plot(AGES[1:], d["alpha_late"][1] * RHO ** (AGES[1:] - 2), "-", color=AXIS, lw=2,
            label=f"decay of the clock eigenvalue, {RHO:.3f} per step")
    ax.plot(AGES[:8], d["alpha_late"], "o", color=RAMP[0], label="α(a) when no runner-up arrived after the maximum", **MK)
    ax.plot(AGES, d["alpha"], "-o", color=BLUE, lw=2, label="α(a), mean over the runner-up's ages", **MK)
    ax.plot(AGES, d["c"], "-o", color=ORANGE, lw=2, label="c(a) of the one-term law  z ∝ M − c S⁺", **MK)
    note(ax, 9.22, d["alpha"][-1], "α", va="center"); note(ax, 9.22, d["c"][-1], "c", va="center")
    same = np.abs(d["alpha"][:4] - d["alpha_late"][:4]).max()                   # the two alpha coincide for a <= 4
    note(ax, 2.35, 1.057, f"a ≤ 4: the light dots lie under the dark ones (equal within {same:.2f})", ha="left", va="center", bbox=BOX)
    ax.set_xlim(0.6, 9.6); ax.set_ylim(0.62, 1.07); ax.set_xticks(AGES)
    ax.set_xlabel("a, steps since the maximum arrived"); ax.set_ylabel("α(a) and c(a)")
    leg(ax, loc="lower left")

    ax = axs[1, 0]
    style(ax, "c  A small gap holds the phase of z back, except one step after a new maximum")
    xc = GAP_EDGES[:-1] + 0.05
    names = ["1 step", "2–4 steps", "5–9 steps"]
    ax.axhline(0, color=AXIS, lw=1)
    for g in (1, 2, 0):
        ax.plot(xc, d["lag"][g], "-o", color=RAMP[g], lw=2, label=names[g], **MK)
        note(ax, -0.012, d["lag"][g][0], names[g], ha="right", va="center")
    note(ax, 0.25, 21.5, "1 step old: no lag, but a scatter to both sides\n"
         f"(median distance from φ(a) in the first bin: {d['absdev'][0, 0]:.0f}°)", ha="left", va="center", linespacing=1.3, bbox=BOX)
    ax.set_xlim(-0.19, 1.02); ax.set_ylim(-2.5, 32); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("relative gap (M − S⁺)/M = 1 − S⁺/M, in bins of 0.1"); ax.set_ylabel("median of φ(a) − arg z (degrees)")
    h, l = ax.get_legend_handles_labels()
    leg(ax, handles=[h[2], h[0], h[1]], labels=[l[2], l[0], l[1]], loc="upper right", title="age of the maximum")

    ax = axs[1, 1]
    tie = [d["rad"][a - 1, -1] for a in RAD_AGES]
    style(ax, "d  The radius follows the gap, but does not vanish at a tie")
    ax.plot([0, 1], [1, 0], "-", color=AXIS, lw=2, label="1 − S⁺/M: radius = gap")
    for g, a in enumerate(RAD_AGES):
        r = d["rad"][a - 1]
        ax.plot(RAD_X[1:], r[1:], "-o", color=RAMP[g], lw=2, label=f"maximum {a} step{'s' * (a > 1)} old", **MK)
        ax.plot(0, r[0], "o", mfc=SURF, mec=RAMP[g], ms=6.5, mew=1.8, zorder=4)
        note(ax, -0.03, r[0], f"{a} step{'s' * (a > 1)}", ha="right", va="center")
    ax.plot([], [], "o", mfc=SURF, mec=MUTED, ms=6.5, mew=1.8, label="states with S ≤ 0 (drawn at 0)")
    ax.annotate("left at a tie (S⁺/M > 0.95):\n" + ", ".join(f"{v:.2f}" for v in tie) + " of w₄M\nat ages " + ", ".join(map(str, RAD_AGES)),
                xy=(0.975, tie[-1] - 0.02), xytext=(0.785, 0.022), fontsize=8.5, color=INK2, ha="right", va="bottom", bbox=BOX,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkB=4))
    ax.set_xlim(-0.17, 1.02); ax.set_ylim(0, 1.06); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("S⁺/M, in bins of 0.1 (0.05 above 0.9)"); ax.set_ylabel("median of |z| / (w₄ M)")
    leg(ax, loc="upper right")

    fig.savefig(OUT / "clock_laws.png", dpi=150)
    plt.close(fig)


def draw_thresholds(d):
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.825, bottom=0.085)
    header(fig, 0.07, "Positions of the five thresholds between the runner-up and the maximum (section 4.1)",
           "f = (θ − S)/(M − S), with M and S the maximum and the runner-up of the inputs before the one that the threshold acts on.\n"
           f"{len(SEEDS) * NB // 10 ** 6}M random sequences, states with S > 0 and M − S > 0.2.   Lines: medians.   Bands: quartiles.")
    f, f4 = d["f"], d["f"][4, :, 1]
    lowest = f[:4, :, 1].min(0)
    style(ax, f"Comparators 1, 6, 8 swing from the runner-up to beyond the maximum; the thresholds of 7 and 4 sag to {f[2, :, 1].min():.2f} and {f4.min():.2f}")
    ax.fill_between(AGES, f4, 1, where=f4 < 1, color=FALSE, alpha=0.13, lw=0)
    for y in (0, 1):
        ax.axhline(y, color=AXIS, lw=1.2)
    names = [f"comparator {j}" for j in COMP] + ["unit 4 (fires above its threshold)"]
    for k in (0, 1, 3, 2, 4):
        ax.fill_between(AGES, f[k, :, 0], f[k, :, 2], color=COLS[k], alpha=0.16, lw=0)
    hs = [ax.plot(AGES, f[k, :, 1], "-o", color=COLS[k], lw=2, label=names[k], zorder=3 + (k == 4), **MK)[0] for k in range(5)]
    hs.append(Patch(color=FALSE, alpha=0.13, lw=0, label="an input here is below the maximum but fires unit 4"))
    for k, (lab, dy, va) in enumerate([("1", 0, "center"), ("6", 0, "center"), ("7", 0.025, "bottom"), ("8", 0, "center"), ("unit 4", -0.03, "top")]):
        note(ax, 9.2, f[k, -1, 1] + dy, lab, va=va)
    note(ax, 10.3, 1.03, "maximum", ha="right", va="bottom"); note(ax, 10.3, -0.03, "runner-up", ha="right", va="top")
    note(ax, 8.42, 0.9, "unit 4 fires\nfalsely here", ha="center", va="center", linespacing=1.15)
    for x0, sel in ((2.5, slice(0, 4)), (6.5, slice(4, 8))):
        note(ax, x0, -0.335, f"ages {sel.start + 1}–{sel.stop}: the lowest of the four comparator medians is "
             f"{lowest[sel].min():.2f} to {lowest[sel].max():.2f}".replace("-", "−"), ha="center", va="center", bbox=BOX)
    ax.set_xlim(0.7, 10.35); ax.set_ylim(-0.42, 1.92); ax.set_xticks(AGES); ax.set_yticks(np.arange(-0.25, 1.76, 0.25))
    ax.set_xlabel("steps since the maximum arrived, at the step the threshold acts")
    ax.set_ylabel("threshold position f = (θ − S)/(M − S)")
    leg(ax, handles=hs, loc="upper left", ncol=2)
    fig.savefig(OUT / "thresholds_by_age.png", dpi=150)
    plt.close(fig)


def draw_level(d):
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 12.1))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.89, bottom=0.055, wspace=0.2, hspace=0.25)
    k2 = float(d["u2_slope"])
    header(fig, 0.07, "Where the maximum is stored: the level coordinate and unit 2 (section 4.1)",
           f"All states with t ≥ 1 of {len(SEEDS) * NB // 10 ** 6}M random sequences.   The maximum $M_{{t-1}}$ and the runner-up "
           "are those of the earlier inputs $x_0\\,..\\,x_{t-1}$, and the age is that of this maximum at step t.\n"
           f"{k2:.1f} is the fitted slope of unit 2 on " + r"$\max(M_{t-1},\,0)$.   d: states with $M_{t-1}$ > 0.3.")
    E = np.linspace(0, HMAX, HBINS + 1)
    vmax = max(d[k].max() for k in ("hist_lev", "hist_young", "hist_old"))
    xl = r"$\max(M_{t-1},\,0)$, the maximum of the earlier inputs"

    def hist(ax, Hc, slope, lab, ylab):
        im = ax.pcolormesh(E, E, np.where(Hc > 0, Hc, np.nan).T, cmap=SEQ, norm=LogNorm(1, vmax), rasterized=True)
        ax.plot([0, HMAX], [0, slope * HMAX], "-", color=ORANGE, lw=1.2, label=lab)
        ax.set_xlim(0, HMAX); ax.set_ylim(0, HMAX); ax.set_aspect("equal")
        ax.set_xlabel(xl); ax.set_ylabel(ylab)
        ax.add_patch(Rectangle((0.555, 0), 0.445, 0.17, transform=ax.transAxes, fc=SURF, ec="none", zorder=0.6))     # hides the grid
        cax = ax.inset_axes([0.60, 0.09, 0.34, 0.03])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[v for v in (1, 1e2, 1e4, 1e6) if v <= vmax])
        cb.outline.set_visible(False); cax.tick_params(colors=INK2, length=0, labelsize=8)
        cax.set_title("states per bin of 0.025 × 0.025", fontsize=8.5, color=INK2, pad=4)

    ax = axs[0, 0]
    style(ax, f"a  The level coordinate is the previous maximum: R² = {d['lev_r2']:.3f}")
    hist(ax, d["hist_lev"], d["lev_slope"], f"{d['lev_slope']:.3f} · x  (least squares through the origin)", r"level coordinate / $w_4$")
    ax.plot([0, HMAX], [0, d["lev_med"][0] * HMAX], "-", color=INK2, lw=1.2,
            label=f"{d['lev_med'][0]:.2f} · x: median one step after a new maximum\n"
                  f"({d['lev_med_s']:.2f} among the states with a runner-up above 0.05)")
    leg(ax, loc="upper left")

    ax = axs[0, 1]
    style(ax, "b  Unit 2 holds it too while the maximum is 1–4 steps old")
    hist(ax, d["hist_young"], 1, "y = x", rf"$h_t[2]$ / {k2:.1f}")
    leg(ax, loc="upper left")

    ax = axs[1, 0]
    style(ax, "c  From 5 steps on it sags below the maximum")
    hist(ax, d["hist_old"], 1, "y = x", rf"$h_t[2]$ / {k2:.1f}")
    lo = d["sag"][6, 0]
    ax.plot([0, HMAX], [0, lo * HMAX], "-", color=INK2, lw=1.2, label=f"{lo:.2f} · x: age 7, no positive runner-up")
    leg(ax, loc="upper left")

    ax = axs[1, 1]
    style(ax, f"d  The sag is towards the runner-up: up to {100 * (1 - d['sag'][4:, 0].min()):.0f}%, none at a tie")
    ax.axhline(1, color=AXIS, lw=1.2)
    for a in range(1, 5):
        ax.plot(SAG_X, d["sag"][a - 1], "-", color=MUTED, lw=1.5, label="1–4 (four lines)" if a == 1 else None)
    for i, a in enumerate(range(5, 10)):
        s = d["sag"][a - 1]
        ax.plot(SAG_X[1:], s[1:], "-o", color=OLD[i], lw=2, label=str(a), **MK)
        ax.plot(0, s[0], "o", mfc=SURF, mec=OLD[i], ms=6.5, mew=1.8, zorder=4)
        note(ax, -0.03, s[0], str(a), ha="right", va="center")
    note(ax, -0.03, d["sag"][:4, 0].mean(), "1–4", ha="right", va="center")
    ax.plot([], [], "o", mfc=SURF, mec=MUTED, ms=6.5, mew=1.8, label="S ≤ 0 (drawn at 0)")
    ax.set_xlim(-0.12, 1.02); ax.set_ylim(0.56, 1.14); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("S⁺/M of the earlier inputs, in bins of 0.1"); ax.set_ylabel(rf"median of $h_t[2]$ / ({k2:.1f} $M_{{t-1}}$)")
    leg(ax, loc="lower right", ncol=2, title="steps since the maximum arrived")
    fig.savefig(OUT / "level_and_unit2.png", dpi=150)
    plt.close(fig)


def report(d):
    r = lambda v, n=2: np.array2string(np.asarray(v, float), precision=n, separator=", ", floatmode="fixed")
    print("--- clock_laws.png (report, section 4.1, in brackets)")
    print("phi(a)            ", r(d["phi"], 1), "[0, 36, 79, 123, 166, 209, 251, 294, 338]")
    print("steps of phi      ", r(np.diff(d["phi"]), 1), "[36, then 42.3 to 43.9]")
    print("half-range over a2", r((d["phi_hi"] - d["phi_lo"]) / 2, 1), "[1.3 to 1.7 for a <= 6, up to 5.5 for a = 8, 9]")
    print("alpha(a)          ", r(d["alpha"]), "[1.01, 1.03, 0.99, 0.95, 0.92, 0.88, 0.85, 0.82, 0.80]")
    print("alpha, a2 > a     ", r(d["alpha_late"]), f"ratio per step from a = 3: {(d['alpha_late'][7] / d['alpha_late'][2]) ** 0.2:.3f} [0.949]")
    print("c(a)              ", r(d["c"]), "[0.87, 0.84, 0.87, 0.86, 0.82, 0.80, 0.77, 0.75, 0.75]")
    for g, (lo, hi) in enumerate(CLASSES):
        print(f"lag, ages {lo}-{hi}     ", r(d["lag"][g], 1))
    print("median |arg z - phi| over a, gap <= .1, .1-.3, .3-.6, > .6:",
          ", ".join(f"{d['absdev'][:, k].min():.1f}-{d['absdev'][:, k].max():.1f}" for k in range(4)), "[18-33, 8-15, 3-7, 0.6-2.6]")
    for a in RAD_AGES:
        print(f"|z|/(w4 M), age {a} ", r(d["rad"][a - 1]), "[tie: 0.21 at age 1, 0.14 at age 8]" if a != 4 else "")
    print("--- thresholds_by_age.png: medians [the report's table is for", int(d["n_ctrl"]), "controlled sequences; recomputed below]")
    for k, j in enumerate(UNITS):
        print(f"unit {j}: random", r(d["f"][k, :, 1]), " controlled", r(d["f_ctrl"][k]))
    print("largest |random median - controlled mean|:", f"{np.abs(d['f'][:, :, 1] - d['f_ctrl']).max():.2f}")
    print("theta_4 / M_old   ", r(d["th4_M"]), "[0.98 young, 0.84 at ages 6-7]")
    print("--- level_and_unit2.png")
    print(f"level/w4 = {d['lev_slope']:.4f} max(M_(t-1), 0), R2 = {d['lev_r2']:.4f} [0.936, 0.993];  median level/(w4 M) by age",
          r(d["lev_med"]), f"[0.98-1.01, then 0.92-0.94];  age 1, states with S > 0.05 only: {d['lev_med_s']:.3f} [0.98-0.99; v_V5: 0.992]")
    print(f"h_t[2] = {d['u2_slope']:.3f} max(M_(t-1), 0), R2 = {d['u2_r2']:.4f} [17.3, 0.974];  median ratio by age", r(d["u2_med"]),
          "[v_V4: 1.02, 1.02, 1.07, 1.05, 0.98, 0.90, 0.85, 0.86]")
    print("h_t[2] ~ a M + b S+, ages 5..9: a", r(d["u2_ab"][4:, 0]), " b", r(d["u2_ab"][4:, 1]),
          " kappa_2 = b/slope", r(d["u2_ab"][4:, 1] / d["u2_slope"]), "[age 7: 10.5 M + 6.9 S+; kappa_2 0.21-0.40]")
    print("ratio at S <= 0 and at S+/M > 0.9, ages 5..8:", r(d["sag"][4:8, 0]), r(d["sag"][4:8, -1]), "[v_V4: 0.86, 0.71, 0.63, 0.66 -> 1.07, 1.04, 1.02, 1.02]")


if __name__ == "__main__":
    data = dict(np.load(CACHE)) if ("--redraw" in sys.argv and CACHE.exists()) else compute()
    OUT.mkdir(exist_ok=True)
    setup()
    draw_clock(data); draw_thresholds(data); draw_level(data)
    report(data)
    print("wrote clock_laws.png, thresholds_by_age.png, level_and_unit2.png to", OUT)
