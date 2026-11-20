"""Attack on I1/I2, part 3: is 'what the comparators hold that hides the max' the clock coordinate?
Same sequences and steps as I1 (t0 = pM + k, no later record), different interventions.  Seeds 5540, 5541.
"""
from v3lib import *

out = Tee("a3_wipe_vs_clock.txt")
x, pM, pS, M, S, nb = gather_I1(500_000, 5540)
out(f"seeds 5540..{5540 + nb - 1}, I1-selected n = {len(x)}")
KS = range(1, 9)
isC = np.isin(ORD_A, COMP)


def table(fn, tmax=7, want_neg=False):
    res = []
    for k in KS:
        v = pM + k <= tmax
        if v.sum() == 0:
            res.append((np.nan, np.nan, np.nan, np.nan)); continue
        h9, hiv = run_iv(x[v], pM[v] + k, fn, return_state=True)
        a = answer(h9)
        res.append((np.mean(a == pM[v]), np.mean(a == pS[v]), np.mean(a == pM[v] + k), np.mean((hiv < -1e-9).any(1))))
    return np.array(res)


def show(label, r):
    out(f"  {label:44s} P(pM): " + " ".join(f"{v:.3f}" for v in r[:, 0]) + "   P(runner-up): " + " ".join(f"{v:.3f}" for v in r[:, 1])
        + "   P(ans = t0): " + " ".join(f"{v:.3f}" for v in r[:, 2]) + "   neg: " + " ".join(f"{v:.2f}" for v in r[:, 3]))


def wipe(h, rows):
    h = h.copy(); h[:, COMP] = 0.0
    return h


def wipe_restore(mode):
    """Wipe the comparators, then put the given mode coordinate back to its pre-wipe value."""
    L, R = LEFT[mode], RIGHT[mode]

    def fn(h, rows):
        z = h[:, ORD_A] @ L
        h = h.copy(); h[:, COMP] = 0.0
        z2 = h[:, ORD_A] @ L
        h[:, ORD_A] += (2 if IS_COMPLEX[mode] else 1) * np.real(np.multiply.outer(z - z2, R))
        return h
    return fn


def comp_only_mode(mode, clip=True):
    """Comparators := their component along the given mode only."""
    L, R = LEFT[mode], RIGHT[mode]

    def fn(h, rows):
        z = h[:, ORD_A] @ L
        d = (2 if IS_COMPLEX[mode] else 1) * np.real(np.multiply.outer(z, R))
        h = h.copy()
        h[:, COMP] = d[:, isC]
        return np.maximum(h, 0.0) if clip else h
    return fn


out("\ncolumns k = 1..8; pooled over t0 = pM + k <= 7.  'neg' = fraction of intervened states with a negative entry")
show("nothing", table(lambda h, rows: h))
show("wipe comparators (I1)", table(wipe))
show("clock mode -> 0 (all 13 units)", table(mode_fn("clock", f=0)))
show("clock mode -> 0, clipped at 0", table(mode_fn("clock", f=0, clip=True)))
show("clock mode -> 0 on comparators only", table(mode_fn("clock", f=0, units=COMP)))
show("clock mode -> 0 on comparators only, clipped", table(mode_fn("clock", f=0, units=COMP, clip=True)))
show("clock mode -> 0 on wave units only", table(mode_fn("clock", f=0, units=WAVE)))
show("clock mode -> 0 on wave units only, clipped", table(mode_fn("clock", f=0, units=WAVE, clip=True)))
show("clock x 0.5", table(mode_fn("clock", f=0.5)))
show("clock x 2", table(mode_fn("clock", f=2.0)))
show("clock x 2, clipped", table(mode_fn("clock", f=2.0, clip=True)))
show("clock rotated k=+1 (no record follows)", table(mode_fn("clock", k=1)))
show("clock rotated k=-1 (no record follows)", table(mode_fn("clock", k=-1)))
show("wipe, then restore the clock coordinate", table(wipe_restore("clock")))
show("comparators := their clock part (clipped)", table(comp_only_mode("clock")))
show("comparators := their clock part (not clipped)", table(comp_only_mode("clock", clip=False)))
for mode in NAMES:
    if mode != "clock":
        show(f"{mode} mode -> 0", table(mode_fn(mode, f=0)))
for mode in NAMES:
    if mode != "clock":
        show(f"wipe, then restore the {mode} coordinate", table(wipe_restore(mode)))

out("\nwhat the wipe does to the mode coordinates (states at t0 = pM + k <= 7): median |coordinate after wipe| / |coordinate before|,")
out("and median |change| / 10.157 (M - S)")
H = run(x)
for k in (1, 2, 3, 4, 5, 6):
    v = pM + k <= 7
    h = H[v, (pM + k)[v]]
    hw = h.copy(); hw[:, COMP] = 0
    c0, c1 = coords(h), coords(hw)
    g = 10.157 * (M[v] - S[v])
    out(f"  k={k}: " + "  ".join(f"{n}: {np.median(np.abs(c1[n]) / np.abs(c0[n])):.2f} ({np.median(np.abs(c1[n] - c0[n]) / g):.2f})" for n in NAMES)
        + f"   | median |z_clock|/(10.157 (M-S)) before = {np.median(np.abs(c0['clock']) / g):.2f}"
        + f"   | |h[COMP]|/|h| = {np.median(np.linalg.norm(h[:, COMP], axis=1) / np.linalg.norm(h, axis=1)):.2f}")

out("\nclock coordinate after the wipe: median |z| / 10.157 (M - S) at t0 + j, j = 0..4 (wiped run | unperturbed run), k = 2, t0 <= 5")
v = pM + 2 <= 5
t0 = (pM + 2)[v]
xv = x[v]
B = len(xv)
h = np.zeros((B, N)); Hw = np.empty((B, T, N))
for t in range(T):
    h = np.maximum(xv[:, t, None] * w + h @ W.T, 0.0)
    rows = np.flatnonzero(t0 == t)
    if rows.size:
        h[rows[:, None], COMP] = 0.0
    Hw[:, t] = h
g = 10.157 * (M[v] - S[v])
r = np.arange(B)
for j in range(5):
    zw = coords(Hw[r, t0 + j])["clock"]; z0 = coords(H[v][r, t0 + j])["clock"]
    out(f"  j={j}: wiped {np.median(np.abs(zw) / g):.3f} | unperturbed {np.median(np.abs(z0) / g):.3f}")

out("\nanswers after scaling the comparators by c = 0.1 (t0 <= 7): pM | runner-up | t0 | t0+1 | other; and mean(ans - pM | other)")
for k in KS:
    v = pM + k <= 7
    if v.sum() == 0:
        continue
    t0 = (pM + k)[v]
    a = answer(run_iv(x[v], t0, scale_units(COMP, 0.1)))
    c1 = a == pM[v]; c2 = a == pS[v]; c3 = (a == t0) & ~c2; c4 = (a == t0 + 1) & ~c2; c5 = ~(c1 | c2 | c3 | c4)
    out(f"  k={k}: {c1.mean():.3f} | {c2.mean():.3f} | {c3.mean():.3f} | {c4.mean():.3f} | {c5.mean():.3f}"
        + (f"   mean(ans - pM | other) = {np.mean((a - pM[v])[c5]):+.2f}; top (ans - pM | other): {np.bincount((a - pM[v])[c5] + 9).argsort()[::-1][:3] - 9}" if c5.any() else ""))
