"""The report's one summary figure: six panels of evidence for the mechanism of M_16,10.

Run:  python -B fig_mechanism.py            (recomputes the data, about 1-2 min, caches fig_mechanism.npz)
      python -B fig_mechanism.py --redraw   (only redraws from the cache)
Writes ../mechanism_summary.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import *
from modes import coords, LAM

CACHE = HIDE / "fig_mechanism.npz"
OMEGA = np.degrees(np.angle(LAM["clock"]))            # 42.75 deg per step
GAP_BINS = [(0.15, 0.30), (0.45, 0.60), (0.85, 1.00)]
V_EDGES = np.array([-1.0, -0.8, -0.6, -0.45, -0.35, -0.25, -0.18, -0.12, -0.07, -0.03, 0.0,
                    0.03, 0.07, 0.12, 0.18, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0])


def compute():
    out = {}
    x = sample(1_000_000, 4242)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    B = len(x)
    ar = np.arange(B)
    M, S, a1 = st["M1"], st["M2"], st["a1"]
    Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)

    # (a) clock coordinate by age of the max, for three bins of the relative gap
    z = coords(H)["clock"] / (10.157 * np.where(M > 0, M, np.nan))
    g = (M - Sp) / np.where(M > 0, M, np.nan)
    za = np.full((len(GAP_BINS), 10), np.nan + 0j)
    for i, (lo, hi) in enumerate(GAP_BINS):
        for a in range(1, 10):
            s = (M > 0.3) & (a1 == a) & (g > lo) & (g <= hi)
            za[i, a] = np.median(z[s].real) + 1j * np.median(z[s].imag)
    out["clock_by_age"] = za

    # (b) thresholds between runner-up (0) and max (1), by age of the max when the next input is compared
    th = np.full((5, 10), np.nan)
    Mo, So, ao = M[:, :-1], S[:, :-1], a1[:, :-1] + 1
    ok = (Mo - So > 0.2) & (So > 0)
    for k, j in enumerate(COMP + [4]):
        theta = -(H[:, :-1] @ W[j]) / w[j]
        pos = (theta - So) / (Mo - So)
        for a in range(1, 10):
            th[k, a] = np.median(pos[ok & (ao == a)])
    out["thresholds"] = th

    # (c) at t = 9: the comparators' vote V and the rest of the readout m, per unit of gap,
    #     as a function of the distance from the max's position (regression on M - S, S, x_9 per age of the max)
    t = 9
    V = H[:, t][:, COMP] @ U[:, COMP].T
    rest = H[:, t] @ U.T - V
    allon = (Z[:, t][:, COMP] > 0).all(1) & (S[:, t] > 0.1)
    A = np.full((2, 10, 10), np.nan)
    for a in range(2, 10):
        s = allon & (a1[:, t] == a)
        X = np.column_stack([(M[:, t] - S[:, t])[s], S[s, t], x[s, t]])
        for k, Y in enumerate((V, rest)):
            A[k, a] = np.linalg.lstsq(X, Y[s], rcond=None)[0][0]
    pairs = [(a, idx) for a in range(2, 10) for idx in range(8)]
    out["vote_d"] = np.array([idx - (9 - a) for a, idx in pairs])
    out["vote_A"] = np.array([A[0, a, idx] for a, idx in pairs])
    out["vote_Am"] = np.array([A[1, a, idx] for a, idx in pairs])

    # (d) provisional scores relative to the runner-up's, in units of M, against (x_p - S)/M
    L = H @ U.T
    med = np.full((3, len(V_EDGES) - 1), np.nan)
    acc_v = [[], [], []]; acc_d = [[], [], []]
    for tt in range(2, 10):
        Mt, St = M[:, tt], S[:, tt]
        ok = St > 0.2
        lS = L[ar, tt, 9 - (tt - st["p2"][:, tt])]
        for p in range(tt + 1):
            a = tt - p
            keep = ok & (st["p2"][:, tt] != p)
            cls = 0 if a == 0 else (1 if a == 1 else 2)
            acc_v[cls].append(((x[:, p] - St) / Mt)[keep])
            acc_d[cls].append(((L[:, tt, 9 - a] - lS) / Mt)[keep])
    for c in range(3):
        v = np.concatenate(acc_v[c]); d = np.concatenate(acc_d[c])
        for b in range(len(V_EDGES) - 1):
            s = (v > V_EDGES[b]) & (v <= V_EDGES[b + 1])
            if s.sum() > 300:
                med[c, b] = np.median(d[s])
    out["score_med"] = med

    # (e) the shift matrix
    Uo = U[:, ORD]
    out["Sigma"] = Uo @ W[np.ix_(ORD, ORD)] @ np.linalg.pinv(Uo)

    # (f) running answer: full readout -> runner-up so far; readout without comparators -> max so far
    Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
    Mall = L - H @ Uc.T
    acc_s = np.full(10, np.nan); acc_m = np.full(10, np.nan)
    for tt in range(1, 10):
        valid = np.arange(10) >= 9 - tt
        sp = S[:, tt] > 0
        ps = np.where(valid, L[:, tt], -np.inf).argmax(1) + tt - 9
        acc_s[tt] = np.mean(ps[sp] == st["p2"][sp, tt])
        if tt >= 2:
            valid_m = valid & (np.arange(10) <= 8)
            ok = sp & (st["p1"][:, tt] < tt)
            pm = np.where(valid_m, Mall[:, tt], -np.inf).argmax(1) + tt - 9
            acc_m[tt] = np.mean(pm[ok] == st["p1"][ok, tt])
    out["acc_s"], out["acc_m"] = acc_s, acc_m
    np.savez(CACHE, **out)
    return out


# ---------------------------------------------------------------- drawing
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]             # ordinal: small, medium, large gap
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ee9a99", "#c03434"])


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.6, **kw)
    return lg


def draw(d):
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF})
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 10.2))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.07, wspace=0.26, hspace=0.36)
    mk = dict(ms=7.5, mec=SURF, mew=1.6)

    # (a)
    ax = axs[0, 0]
    style(ax, "a  The clock: direction = age of the max, radius grows with the gap")
    for i, (lo, hi) in enumerate(GAP_BINS):
        zz = d["clock_by_age"][i, 1:]
        ax.plot(zz.real, zz.imag, "-o", color=RAMP[i], lw=2, label=f"{lo:.2f}–{hi:.2f}", **mk)
    zz = d["clock_by_age"][2, 1:]
    for a, c in zip(range(1, 10), zz):
        ax.annotate(str(a), (c.real, c.imag), xytext=(c.real * 1.16, c.imag * 1.16), ha="center", va="center",
                    fontsize=8.5, color=INK2)
    ax.plot([0], [0], "+", color=INK2, ms=9, mew=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Re z / (10.157 M)"); ax.set_ylabel("Im z / (10.157 M)")
    ax.text(0.98, 0.03, "numbers: steps since the max arrived", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=INK2)
    lg = legend(ax, loc="lower left", title="gap (M − S⁺)/M")
    lg.get_title().set_color(INK2); lg.get_title().set_fontsize(8.5)

    # (b)
    ax = axs[0, 1]
    style(ax, "b  The comparators' thresholds swing with it")
    ages = np.arange(1, 10)
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for k, j in enumerate(COMP):
        ax.plot(ages, d["thresholds"][k, 1:], "-o", color=cols[k], lw=2, label=f"comparator {j}", **mk)
    ax.plot(ages, d["thresholds"][4, 1:], "-", color=MUTED, lw=2, label="unit 4 (fires above it)")
    ax.axhline(0, color=AXIS, lw=1); ax.axhline(1, color=AXIS, lw=1)
    ax.text(10.55, -0.03, "runner-up", va="top", ha="right", fontsize=8.5, color=INK2)
    ax.text(10.55, 1.03, "max", va="bottom", ha="right", fontsize=8.5, color=INK2)
    for k, j in enumerate(COMP):
        ax.text(9.2, d["thresholds"][k, 9], f"{j}", va="center", fontsize=8.5, color=INK2)
    ax.text(9.2, d["thresholds"][4, 9] - 0.07, "unit 4", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0.7, 10.6); ax.set_xticks(ages)
    ax.set_xlabel("steps since the max arrived"); ax.set_ylabel("threshold position, (θ − S)/(M − S)")
    legend(ax, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(-0.25, 2.05)

    # (c)
    ax = axs[0, 2]
    style(ax, "c  In the readout: two waves on the max's position that cancel")
    dd = np.linspace(-7, 7, 300)
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(dd, -66 * np.cos(np.radians(OMEGA * dd)), "-", color=AXIS, lw=2, label="∓66 cos(42.75° · d)")
    ax.plot(dd, 66 * np.cos(np.radians(OMEGA * dd)), "-", color=AXIS, lw=2)
    ax.plot(d["vote_d"], d["vote_A"], "o", color=BLUE, label="V: through the comparators", **mk)
    ax.plot(d["vote_d"], d["vote_Am"], "o", color=ORANGE, label="m: through the other units", **mk)
    ds = np.arange(-7, 8)
    tot = [np.mean((d["vote_A"] + d["vote_Am"])[d["vote_d"] == k]) for k in ds]
    ax.plot(ds, tot, "-", color=INK, lw=2, label="V + m (mean over ages of the max)")
    ax.set_xlabel("d = position − position of the max"); ax.set_ylabel("logits per unit of (M − S)")
    ax.set_xticks(range(-7, 8)); ax.set_ylim(-110, 150)
    legend(ax, loc="upper center", ncol=2)

    # (d)
    ax = axs[1, 0]
    style(ax, "d  Scores against the runner-up's: linear when new, clipped later")
    vc = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
    for c, (col, lab) in enumerate(zip([BLUE, ORANGE, AQUA], ["input arriving now", "one step old", "two or more steps old"])):
        ax.plot(vc, d["score_med"][c], "-o", color=col, lw=2, label=lab, **mk)
    ax.axvline(0, color=AXIS, lw=1); ax.axhline(0, color=AXIS, lw=1)
    ax.plot([-1, 0], [-61.6, 0], "-", color=MUTED, lw=1)
    ax.text(-0.78, -57, "slope 61.6", fontsize=8.5, color=INK2, rotation=0, ha="left")
    ax.text(-0.5, 3.5, "below the runner-up", fontsize=8.5, color=INK2, ha="center")
    ax.text(0.5, 3.5, "the max", fontsize=8.5, color=INK2, ha="center")
    ax.set_xlabel("(input − runner-up) / max"); ax.set_ylabel("score − runner-up's score, in units of the max")
    ax.set_xlim(-1.02, 1.3); ax.set_ylim(-66, 8)
    for c, lab in enumerate(["arriving now", "one step old", "older"]):
        ax.text(0.94, d["score_med"][c][-1], lab, va="center", fontsize=8.5, color=INK2)
    legend(ax, loc="lower center", bbox_to_anchor=(0.42, 0.0))

    # (e)
    ax = axs[1, 1]
    style(ax, "e  The resting map moves each score one position (matrix Σ)")
    ax.grid(False)
    Sg = d["Sigma"][:9]                                  # row 9 rides on the readout's smallest singular direction
    im = ax.imshow(Sg, cmap=DIV, norm=TwoSlopeNorm(0, -1.05, 1.05), aspect="equal")
    for i in range(9):
        for j in range(10):
            if abs(Sg[i, j]) >= 0.3:
                ax.text(j, i, f"{Sg[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(Sg[i, j]) > 0.75 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(9))
    ax.set_xlabel("logit index before the step (9 = newest input)"); ax.set_ylabel("logit index after the step")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, length=0, labelsize=8)

    # (f)
    ax = axs[1, 2]
    style(ax, "f  Read at any step, the state already holds the answer so far")
    tt = np.arange(1, 10)
    ax.plot(tt, d["acc_s"][1:], "-o", color=BLUE, lw=2, label="all units: top score = second largest so far", **mk)
    ax.plot(tt[1:], d["acc_m"][2:], "-o", color=ORANGE, lw=2,
            label="without the comparators, among older inputs\n(max not the current input): top score = largest so far", **mk)
    ax.set_ylim(0.86, 1.005); ax.set_xticks(tt)
    ax.set_xlabel("step t at which the final readout is applied")
    ax.set_ylabel("fraction of states with a positive runner-up")
    legend(ax, loc="lower left")

    fig.savefig(OUT / "mechanism_summary.png", dpi=150)
    print("wrote", OUT / "mechanism_summary.png")


if __name__ == "__main__":
    data = dict(np.load(CACHE)) if ("--redraw" in sys.argv and CACHE.exists()) else compute()
    draw(data)
