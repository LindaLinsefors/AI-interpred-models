"""Three figures on the interventions of section 3.5 and on the feedback experiment e29 of section 4.1.

Run:  python -B plot_interventions.py            (recomputes, about 1 min, caches plot_interventions.npz)
      python -B plot_interventions.py --redraw   (only redraws from the cache)
Writes ../plots/intervention_wipe.png, ../plots/intervention_clock.png, ../plots/feedback_by_mode.png
and prints the plotted numbers.  Inputs are iid N(0,1): 1M sequences for the wipe and for the feedback, 2M
for the clock (of which 10% are selected).  The seeds differ from those of e17, e18 and e29.

intervention_wipe.png (as e17).  Sequences with a final runner-up above 0.2 and a final gap above 0.2.  At step
    t0 = p_M + k (p_M: position of the final maximum, k: its age at that step) the four comparators are set
    to zero once, h_t0[1,6,7,8] = 0, and the run continues with the real inputs.
    a: P(answer = p_M) for every age k and wipe step t0.
    b: where the answer goes, pooled over the wipes at t0 <= 7, with and without the wipe.

intervention_clock.png (as e18).  Sequences in which the final maximum arrives at step t_r = 3..8, the old
    maximum is at least two steps old at step t_r - 1 and ends as the runner-up, and the old runner-up (above
    0.2, at least 0.1 below the old maximum) ends third.  The state after step t_r - 1 is changed once:
    h[o] -> h[o] - (1 - f) 2 Re(z r_clock), on all 13 non-pulse units o or on some of them (not clipped).
    a: where the answer goes for f = 0, 0.25, 0.5, 0.75, 1 on all 13 units.
    b: P(answer = old maximum) unperturbed, with the clock mode removed (f = 0), with only its entries on
       the nine wave units removed, and with only its entries on the four comparators removed; next to each,
       the median of |z| after the change relative to |z| before it.

feedback_by_mode.png (as e29).  The eight wave units other than unit 2 add eps_j = sum_u W[j,u] h[u] to the
    pre-activation of comparator j.  h[o] is an exact sum of nine mode patterns, so eps is a sum of nine parts.
    a: accuracy with one part removed at every step, against the rms difference that part makes between the
       four thresholds (part / g_j, in units of the running maximum M; states h_0..h_8 with M > 0.3 and a
       maximum at least one step old).
    b: the same accuracies as bars, next to the accuracy with only that part kept.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import *
from modes import NAMES, LEFT, RIGHT, IS_COMPLEX
from fig_mechanism import style, legend, SURF, INK, INK2, MUTED, AXIS, BLUE, ORANGE, AQUA

HIDE = Path(__file__).resolve().parent
OUT = HIDE.parent / "plots"
CACHE = HIDE / "plot_interventions.npz"
SAT = [0, 3, 10, 11, 12, 13, 14, 15]                    # the wave units other than unit 2
F = [0, 0.25, 0.5, 0.75, 1]                             # factors on the clock mode
FULL = dict(level="level", runner="runner", clock="clock", h2="second harmonic", alt="alternating",
            h3="third harmonic", short="short clock", trans="transient", echo="echo")


def answers(x, t0=None, edit=None, Wm=W):
    """Final answers of the network with recurrent weights Wm.  With edit, the state of row b is replaced
    by edit(h, rows) right after step t0[b]."""
    h = np.zeros((len(x), N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ Wm.T, 0)
        if edit is not None:
            rows = np.flatnonzero(t0 == t)
            if rows.size:
                h[rows] = edit(h[rows], rows)
    return (h @ U.T).argmax(1)


# ---------------------------------------------------------------- computing
def compute_wipe(n=1_000_000, seed=2101):
    x = sample(n, seed)
    order = np.argsort(x, axis=1)
    pM, pS = order[:, -1], order[:, -2]
    ar = np.arange(n)
    keep = (x[ar, pS] > 0.2) & (x[ar, pM] - x[ar, pS] > 0.2)
    x, pM, pS = x[keep], pM[keep], pS[keep]

    def wipe(h, rows):
        h[:, COMP] = 0
        return h

    base = answers(x)
    cnt = np.zeros((2, 10, 10, 5), int)                 # no wipe / wipe, age k, wipe step t0, outcome
    for k in range(1, 10):
        ok = pM + k <= 9
        t0, m, s = (pM + k)[ok], pM[ok], pS[ok]
        for i, a in enumerate((base[ok], answers(x[ok], t0, wipe))):
            # 0 the maximum, 1 the runner-up, 2 the runner-up when it is the wipe-step input,
            # 3 the wipe-step input (not the runner-up), 4 another position
            code = np.select([a == m, (a == s) & (s != t0), a == s, a == t0], [0, 1, 2, 3], 4)
            cnt[i, k] = np.bincount(t0 * 5 + code, minlength=50).reshape(10, 5)
    return dict(wipe_cnt=cnt, wipe_n=np.array([n, keep.sum()]), wipe_base=np.mean(base == pS))


def scale_clock(h, f, units):
    """Clock pattern times f on the listed units: h[u] -> h[u] - (1 - f) [2 Re(z r_clock)][u]."""
    z = h[:, ORD] @ LEFT["clock"]
    h = h.copy()
    h[:, ORD] -= (1 - f) * 2 * np.real(np.outer(z, RIGHT["clock"])) * np.isin(ORD, units)
    return h


def compute_clock(n=2_000_000, seed=2102):                      # one sequence in ten is selected
    x = sample(n, seed)
    order = np.argsort(x, axis=1)
    tr = order[:, -1]                                           # step of the final maximum
    xp = np.where(np.arange(T) < tr[:, None], x, -np.inf)       # the inputs before it
    o2 = np.argsort(xp, axis=1)
    p0, q0 = o2[:, -1], o2[:, -2]                               # old maximum, old runner-up
    ar = np.arange(n)
    M0, S0 = xp[ar, p0], xp[ar, q0]
    with np.errstate(invalid="ignore"):                         # -inf - -inf where nothing came before
        ok = ((tr >= 3) & (tr <= 8) & (tr - 1 - p0 >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1)
              & (order[:, -2] == p0) & (order[:, -3] == q0))
    x, tr, p0, q0 = x[ok], tr[ok], p0[ok], q0[ok]

    hpre = np.empty((len(x), N))                                # the unperturbed state after step t_r - 1

    def grab(h, rows):
        hpre[rows] = h
        return h

    answers(x, tr - 1, grab)
    zpre = np.abs(hpre[:, ORD] @ LEFT["clock"])
    cases = [(f, ORD) for f in F] + [(0, WAVE), (0, COMP)]
    shares, zleft = [], []
    for f, units in cases:
        a = answers(x, tr - 1, lambda h, rows: scale_clock(h, f, units))
        hit = [a == p0, a == tr, a == q0]                       # three different positions
        shares.append([np.mean(v) for v in hit] + [1 - np.mean(np.any(hit, 0))])
        zleft.append(np.median(np.abs(scale_clock(hpre, f, units)[:, ORD] @ LEFT["clock"]) / zpre))
    return dict(clock_out=np.array(shares), clock_z=np.array(zleft), clock_n=np.array([n, len(x)]))


def compute_feedback(n=1_000_000, seed=2103):
    sat = [ORD.index(u) for u in SAT]
    P = [(2 if IS_COMPLEX[m] else 1) * np.outer(RIGHT[m], LEFT[m]).real for m in NAMES]   # h[o] = sum_m P[m] h[o]
    assert np.abs(sum(P) - np.eye(13)).max() < 1e-9
    G = np.array([W[np.ix_(COMP, SAT)] @ p[sat] for p in P])    # (9, 4, 13): mode m's part of eps is G[m] h[o]
    x = sample(n, seed)
    y = second_argmax(x)

    def acc(D):
        """Accuracy when D h[o] is taken out of the comparators' pre-activations at every step."""
        Wm = W.copy()
        Wm[np.ix_(COMP, ORD)] -= D
        return np.mean(answers(x, Wm=Wm) == y)

    out = dict(fb_intact=acc(0 * G[0]), fb_none=acc(G.sum(0)), fb_removed=np.array([acc(Gm) for Gm in G]),
               fb_alone=np.array([acc(G.sum(0) - Gm) for Gm in G]), fb_n=n)

    # Size of each part in threshold units, relative to the running maximum M.  With u = h[o] / M, mode m's part
    # of threshold j is A[m, j] . u, so its mean and rms over the states follow from <u> and <u u^T>.
    A = G / -w[COMP, None]                                      # (9, 4, 13)
    C = A.mean(1)                                               # the part common to the four thresholds
    M = np.maximum.accumulate(x, 1)
    h, u1, u2, cnt = np.zeros((n, N)), 0, 0, 0
    for t in range(T - 1):                                      # states h_0..h_8 of the intact network
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        ok = (M[:, t] > 0.3) & (x[:, t] < M[:, t])              # maximum above 0.3 and at least one step old
        u = h[ok][:, ORD] / M[ok, t, None]
        u1, u2, cnt = u1 + u.sum(0), u2 + u.T @ u, cnt + len(u)

    def ms(V):                                                  # mean square of V . u over the states
        return np.einsum("...i,ij,...j->...", V, u2 / cnt, V)

    out.update(fb_mean=C @ u1 / cnt, fb_rms_common=np.sqrt(ms(C)), fb_rms_diff=np.sqrt(ms(A - C[:, None]).mean(1)))
    return out


# ---------------------------------------------------------------- drawing
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])


def ink_on(rgb):
    """White or near-black text, whichever contrasts more with the fill."""
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb[:3]]
    return "#ffffff" if np.dot([0.2126, 0.7152, 0.0722], lin) < 0.18 else INK


def footer(fig, text):
    fig.text(0.012, 0.012, text, fontsize=8.5, color=INK2, ha="left", va="bottom", linespacing=1.4)


def draw_wipe(d):
    c = d["wipe_cnt"].astype(float)                     # (no wipe / wipe, age k, wipe step t0, outcome)
    n_all, n_sel = d["wipe_n"]
    fig = plt.figure(figsize=(13.2, 6.3))
    ax, bx = fig.add_axes([0.045, 0.185, 0.36, 0.74]), fig.add_axes([0.565, 0.185, 0.42, 0.74])

    # a: P(answer = maximum) by age of the maximum and step of the wipe
    style(ax, "a  Comparators set to zero once: the network then answers the maximum")
    ax.grid(False)
    with np.errstate(invalid="ignore"):
        P = c[1, 1:, 1:, 0] / c[1, 1:, 1:].sum(-1)
    im = ax.pcolormesh(np.arange(0.5, 10), np.arange(0.5, 10), np.ma.masked_invalid(P), cmap=SEQ, vmin=0, vmax=1,
                       edgecolors=SURF, linewidth=2)
    for k in range(1, 10):
        for t0 in range(k, 10):
            p = P[k - 1, t0 - 1]
            ax.text(t0, k, f"{p:.3f}", ha="center", va="center", fontsize=8.5, color=ink_on(SEQ(p)))
    ax.set_aspect("equal"); ax.set_anchor("W")
    ax.set_xlim(0.5, 9.5); ax.set_ylim(9.5, 0.5)
    ax.set_xticks(range(1, 10)); ax.set_yticks(range(1, 10))
    ax.set_xlabel("step of the wipe, t₀ (the last step is 9)")
    ax.set_ylabel("age k of the maximum at the wipe (steps)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0, 0.04, 1]))
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, length=0, labelsize=8)
    cb.set_label("P(answer = position of the maximum)", color=INK2, fontsize=9)
    miss8 = c[1, :, 8, 2:4].sum() / c[1, :, 8, 1:].sum()           # at t0 = 8: share of the other answers that are position 8
    k9 = [1, 2, 3, 9]                                              # the ages with a light cell in the last column
    at9 = c[1, k9, 9, 2:4].sum(1) / c[1, k9, 9].sum(1)             # at t0 = 9: P(answer = position 9) for these k
    ax.text(0.62, 7.75, "No cells below the diagonal: t₀ = position of\nthe maximum + k.\n\n"
            f"Wipe at step 8: {miss8:.0%} of the answers that are\nnot the maximum are position 8, the input\nof the wipe step.\n\n"
            f"Wipe at step 9: V is missing from the readout\nitself. For k ≤ 3 and k = 9 the answer is\n"
            f"position 9 ({at9.min():.0%}–{at9.max():.0%}).", fontsize=8.5, color=INK2, ha="left", va="center", linespacing=1.3)

    # b: where the answer goes, pooled over the wipes at t0 <= 7
    style(bx, "b  Wipes at steps 1–7: the answer flips from runner-up to maximum")
    bx.grid(axis="y", visible=False)
    q = c[:, :, 1:8].sum((1, 2))
    n_runs = q[1].sum()
    q = np.column_stack([q[:, 0], q[:, 1] + q[:, 2], q[:, 3], q[:, 4]]) / q.sum(1, keepdims=True)
    yy, bh = np.arange(4), 0.2
    for i, (col, lab) in enumerate(zip([MUTED, BLUE], ["no wipe", "comparators set to zero at step t₀"])):
        pos = yy + (i - 0.5) * (bh + 0.04)
        bx.barh(pos, q[i], bh, color=col, label=lab, zorder=3)
        for y0, v in zip(pos, q[i]):
            bx.text(v + 0.01, y0, f"{v:.3f}", ha="left", va="center", fontsize=8.5, color=INK2)
        bx.text(0.012, pos[q[i].argmax()], lab, ha="left", va="center_baseline", fontsize=8.5, color="#ffffff", zorder=4)
    bx.set_yticks(yy, ["the maximum", "the runner-up", "the input of\nthe wipe step", "another\nposition"])
    bx.tick_params(axis="y", labelsize=9)
    bx.set_ylim(3.6, -0.6); bx.set_xlim(0, 1.1); bx.set_xticks(np.arange(0, 1.01, 0.2))
    bx.set_xlabel("share of runs in which the network gives this answer (wipe at a step t₀ ≤ 7)")
    legend(bx, loc="lower right")

    footer(fig, f"Sequences with a final runner-up above 0.2 and a final gap above 0.2 ({n_sel / n_all:.0%} of {n_all:,} N(0,1) sequences). "
           "The wipe sets units 1, 6, 7, 8 to zero once, right after step t₀ = position of the maximum + k,\n"
           "and the run continues with the real inputs. Every sequence is used once for each possible t₀. "
           f"Panel b pools the {int(n_runs):,} runs with t₀ ≤ 7 and shows the same runs without the wipe (counted once per\n"
           f"sequence, the network answers the runner-up in {float(d['wipe_base']):.1%}). "
           "A runner-up that is itself the input of the wipe step counts as runner-up.")
    fig.savefig(OUT / "intervention_wipe.png", dpi=150)
    print("wrote", OUT / "intervention_wipe.png")
    print("  P(max) pooled over t0 <= 7, k = 1..7:", np.round(c[1, 1:8, 1:8, 0].sum(1) / c[1, 1:8, 1:8].sum((1, 2)), 4))
    print("  P(max) at t0 = 8, k = 1..8:", np.round(P[:8, 7], 4))
    print("  t0 <= 7 pooled [max, runner-up, wipe-step input, other]: no wipe", np.round(q[0], 4), " wipe", np.round(q[1], 4))
    print(f"  selected {n_sel} of {n_all}; unperturbed P(runner-up) = {float(d['wipe_base']):.4f}; "
          f"t0 = 8: {miss8:.4f} of the non-max answers are position 8; t0 = 9, k = 1, 2, 3, 9: P(position 9) = {np.round(at9, 4)}")


def draw_clock(d):
    S, zl = d["clock_out"], d["clock_z"]                # rows: f = 0, .25, .5, .75, 1 on all 13 units; wave units only; comparators only
    n_all, n_sel = d["clock_n"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 6.3))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.925, bottom=0.215, wspace=0.16)

    # a: where the answer goes against the factor on the clock mode
    style(ax, "a  With the clock mode shrunk, the old maximum is not promoted")
    ax.grid(axis="x", visible=False)
    cols = [BLUE, ORANGE, AQUA, AXIS]
    labs = ["the old maximum (correct)", "the new maximum", "the old runner-up", f"another position (at most {S[:5, 3].max():.2f})"]
    bottom = np.zeros(5)
    for j in range(4):
        ax.bar(range(5), S[:5, j], 0.42, bottom=bottom, color=cols[j], edgecolor=SURF, linewidth=1.5, label=labs[j], zorder=3)
        for i in range(5):
            if j < 3 and S[i, j] >= 0.045:
                ax.text(i, bottom[i] + S[i, j] / 2, f"{S[i, j]:.3f}", ha="center", va="center", fontsize=8.5, color="#ffffff")
        bottom += S[:5, j]
    mid = np.cumsum(S[0]) - S[0] / 2                            # direct labels on the first bar
    box = dict(facecolor=SURF, edgecolor="none", pad=1.5)       # hides the gridline behind a label
    for j in (1, 2, 3):
        ax.text(-0.27, mid[j], labs[j].split(" (")[0], ha="right", va="center", fontsize=8.5, color=INK2, bbox=box, zorder=2)
    ax.annotate(f"the old maximum\n{S[0, 0]:.3f}", (-0.22, mid[0]), xytext=(-0.42, 0.09), fontsize=8.5, color=INK2, ha="right", va="center",
                linespacing=1.3, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=2, shrinkB=0))
    ax.set_xticks(range(5), ["0\n(mode removed)", "0.25", "0.5", "0.75", "1\n(unperturbed)"])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-1.5, 4.5); ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("factor f on the clock mode, one step before the new maximum", x=3.5 / 6)   # centred under the bars
    ax.set_ylabel("share of sequences")
    lg = legend(ax, loc="upper left", ncol=2, title="the network answers")
    lg.get_title().set_color(INK2); lg.get_title().set_fontsize(8.5); lg.set_alignment("left")

    # b: the dissociation
    style(bx, "b  The promotion needs the wave units' share of the clock pattern, not z")
    bx.grid(axis="y", visible=False)
    rows = [("unperturbed", 4), ("clock mode removed on all 13 units (f = 0 in panel a)", 0),
            ("only the wave units' share removed (units 0, 2, 3, 10–15)", 5),
            ("only the comparators' share removed (units 1, 6, 7, 8)", 6)]
    for y, (lab, i) in enumerate(rows):
        note = "z unchanged" if abs(zl[i] - 1) < 1e-9 else "z = 0" if zl[i] < 1e-9 else f"z keeps {zl[i]:.0%} of its size (median)"
        bx.barh(y, S[i, 0], 0.17, color=BLUE, zorder=3)
        bx.text(S[i, 0] + 0.012, y, f"{S[i, 0]:.3f}", va="center", ha="left", fontsize=9, color=INK)
        bx.text(0.004, y - 0.16, lab, va="bottom", ha="left", fontsize=9, color=INK, bbox=box, zorder=2)
        bx.text(1.115, y - 0.16, note, va="bottom", ha="right", fontsize=8.5, color=INK2, bbox=box, zorder=2)
    bx.set_ylim(3.4, -0.7); bx.set_yticks([])
    bx.set_xlim(0, 1.12); bx.set_xticks(np.arange(0, 1.01, 0.2))
    bx.set_xlabel("P(the network answers the old maximum)")

    footer(fig, "Sequences in which the final maximum arrives at step 3 to 8, the old maximum is at least two steps old at the step before and ends as the runner-up, "
           "and the old runner-up (above 0.2, at least 0.1\n"
           f"below the old maximum) ends third: {n_sel:,} of {n_all:,} N(0,1) sequences ({n_sel / n_all:.0%}). "
           "At the step before the new maximum the state is changed once:  h[o] → h[o] − (1 − f) · 2 Re(z r),  with o the 13 non-pulse\n"
           "units, z the clock coordinate and 2 Re(z r) the clock pattern (section 2 of the report). "
           "Panel b applies the change with f = 0 to all 13 units, to the nine wave units only, or to the four comparators only.")
    fig.savefig(OUT / "intervention_clock.png", dpi=150)
    print("wrote", OUT / "intervention_clock.png")
    for f, s in zip(F, S):
        print(f"  f = {f:<4}: P(old max) {s[0]:.4f}  P(new max) {s[1]:.4f}  P(old runner-up) {s[2]:.4f}  P(other) {s[3]:.4f}")
    for lab, i in (("wave units only", 5), ("comparators only", 6)):
        print(f"  f = 0, {lab}: P(old max) {S[i, 0]:.4f}  P(new max) {S[i, 1]:.4f}  P(old runner-up) {S[i, 2]:.4f}  median |z'|/|z| {zl[i]:.3f}")
    print(f"  selected {n_sel} of {n_all}")


def draw_feedback(d):
    acc, alone, dif = d["fb_removed"], d["fb_alone"], d["fb_rms_diff"]
    intact, none = float(d["fb_intact"]), float(d["fb_none"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 6.3))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.215, wspace=0.3)

    # a: accuracy without a part against the differences it makes between the thresholds
    style(ax, "a  Parts that make larger threshold differences cost more to remove")
    for v, lab in ((intact, "intact network"), (none, "all nine parts removed")):
        ax.axhline(v, color=AXIS, lw=1.2, zorder=1)
        ax.text(0.0425, v + 0.006, f"{lab}: {v:.3f}", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.plot(dif, acc, "o", color=BLUE, ms=8.5, mec=SURF, mew=1.6, zorder=3)
    off = dict(level=(-9, 0, "right"), trans=(9, 0, "left"), runner=(9, -9, "left"), short=(-9, -9, "right"), h3=(9, 0, "left"),
               h2=(9, 0, "left"), clock=(9, -5, "left"), echo=(9, -4, "left"), alt=(9, -3, "left"))
    for m, x0, y0 in zip(NAMES, dif, acc):
        dx, dy, ha = off[m]
        ax.annotate(FULL[m], (x0, y0), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 0.043); ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", pad=7)                             # keeps the tick labels 0.000 and 0.5 apart at the origin
    ax.set_xlabel("rms difference the part makes between the four thresholds (units of the maximum M)")
    ax.set_ylabel("accuracy with the part removed at every step")

    # b: the accuracies as bars
    style(bx, "b  The level's part costs most, and no part is enough on its own")
    bx.grid(axis="y", visible=False)
    order = np.argsort(acc)
    yy, bh = np.arange(9), 0.34
    for i, (vals, col, lab) in enumerate(((acc, BLUE, "this part removed"), (alone, ORANGE, "only this part kept"))):
        pos = yy + (i - 0.5) * (bh + 0.04)
        bx.barh(pos, vals[order], bh, color=col, label=lab, zorder=3)
        for y0, v in zip(pos, vals[order]):
            bx.text(v - 0.008, y0, f"{v:.3f}", va="center_baseline", ha="right", fontsize=8, color="#ffffff", zorder=4)
        bx.text(0.012, pos[0], lab, va="center_baseline", ha="left", fontsize=8, color="#ffffff", zorder=4)   # direct label, first row
    for v, lab in ((intact, "intact"), (none, "all nine removed")):
        bx.axvline(v, color=AXIS, lw=1.2, zorder=1)
        bx.text(v - 0.008, -0.92, f"{lab}: {v:.3f}", ha="right", va="center", fontsize=8.5, color=INK2)
    bx.set_yticks(yy, [FULL[NAMES[k]] for k in order])
    bx.tick_params(axis="y", labelsize=9)
    bx.set_ylim(9.95, -1.4); bx.set_xlim(0, 1.06); bx.set_xticks(np.arange(0, 1.01, 0.2))
    bx.set_xlabel("accuracy of the network")
    legend(bx, loc="lower left", ncol=1)

    footer(fig, "Feedback: what the eight wave units other than unit 2 add to the pre-activations of the four comparators, Σ W[j, u] h[u] over those units u. "
           "The state of the 13 non-pulse units is an exact sum of nine mode patterns,\nso the feedback is a sum of nine parts. "
           "A part is removed, or kept alone, at every step. Its size is in threshold units (divided by the comparator's input gain −w[j]) "
           "and relative to the running maximum M, over states\nwith M > 0.3 and a maximum at least one step old; "
           f"the difference is the deviation from the mean of the four thresholds. {int(d['fb_n']):,} N(0,1) sequences "
           "(a new sample: last digits can differ from the table in section 4.1).")
    fig.savefig(OUT / "feedback_by_mode.png", dpi=150)
    print("wrote", OUT / "feedback_by_mode.png")
    print(f"  intact {intact:.4f}   all nine parts removed {none:.4f}")
    print("  mode      removed   kept alone   mean common part   rms common part   rms difference   (units of M)")
    for k in order:
        print(f"  {NAMES[k]:8s}  {acc[k]:.4f}    {alone[k]:.4f}        {d['fb_mean'][k]:+.4f}           {d['fb_rms_common'][k]:.4f}           {dif[k]:.4f}")


if __name__ == "__main__":
    if "--redraw" in sys.argv and CACHE.exists():
        data = dict(np.load(CACHE))
    else:
        data = {**compute_wipe(), **compute_clock(), **compute_feedback()}
        np.savez(CACHE, **data)
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF})
    draw_wipe(data)
    draw_clock(data)
    draw_feedback(data)
