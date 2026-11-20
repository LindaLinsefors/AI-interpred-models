"""Last controls.  Seeds 5900.. (I2 pool), 5920.. (I1 pool), rng 5999 for the random directions / donors.

 1. I2: random perturbations of several sizes (calibration for 'clock' 0.335 and 'runner' 0.464 of |h[ORD]|)
 2. I2: direction check that does not use any intervention - phase of the clock coordinate vs age of the max
 3. I1: on-manifold 'transplant' - give the comparators the values they have in ANOTHER sequence
        (same age of the max / different age), rescaled by the ratio of the maxima
"""
from v3lib import *

out = Tee("a7_controls.txt")
rng = np.random.default_rng(5999)

# ------------------------------------------------------------------ 1
x, tr, p0, q0, M0, S0, nb = gather_I2(250_000, 5900)
n = len(x); r = np.arange(n); tint = tr - 1
out(f"[1] I2 pool: seeds 5900..{5900 + nb - 1}, n = {n}")
H0 = run(x)
hpre = H0[r, tint]
hn = np.linalg.norm(hpre[:, ORD_A], axis=1)


def add_fn(delta, clip=True):
    def fn(h, rows):
        h = h.copy()
        h[:, ORD_A] += delta[rows]
        return np.maximum(h, 0.0) if clip else h
    return fn


out("    random Gaussian direction on the 13 ordinary units, |delta| = size * |h[ORD]|, clipped at 0:")
for size in (0.05, 0.1, 0.2, 0.335, 0.464, 0.7, 1.0):
    g = rng.standard_normal((n, 13)); g *= (size * hn / np.linalg.norm(g, axis=1))[:, None]
    a = answer(run_iv(x, tint, add_fn(g)))
    out(f"    size {size:<5}: P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}")
out("    same, wave units only:")
isW = np.isin(ORD_A, WAVE)
for size in (0.1, 0.2, 0.335, 0.464):
    g = rng.standard_normal((n, 13)) * isW; g *= (size * hn / np.linalg.norm(g, axis=1))[:, None]
    a = answer(run_iv(x, tint, add_fn(g)))
    out(f"    size {size:<5}: P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}")
for mode in ("clock", "runner", "level", "h2", "short"):
    z = hpre[:, ORD_A] @ LEFT[mode]
    d = (2 if IS_COMPLEX[mode] else 1) * np.real(np.multiply.outer(z, RIGHT[mode]))
    out(f"    removal of the {mode:6s} mode: median |delta|/|h[ORD]| = {np.median(np.linalg.norm(d, axis=1) / hn):.3f}; on wave units only "
        f"{np.median(np.linalg.norm(d * isW, axis=1) / hn):.3f}; on comparators only {np.median(np.linalg.norm(d * ~isW, axis=1) / hn):.3f}")

# ------------------------------------------------------------------ 2
out("\n[2] no intervention: clock coordinate at t_r - 1 by age a of the old max: phase of the median of z / (10.157 (M0 - S0)), and its modulus")
age = tr - 1 - p0
zc = (hpre[:, ORD_A] @ LEFT["clock"]) / (10.157 * (M0 - S0))
prev = None
for a_ in range(2, 8):
    m = (age == a_) & (M0 - S0 > 0.3)
    med = np.median(zc[m].real) + 1j * np.median(zc[m].imag)
    ph = np.degrees(np.angle(med)) % 360
    out(f"    age {a_}: n = {m.sum():6d}  phase = {ph:6.1f} deg  modulus = {abs(med):.3f}" + (f"   step from previous age: {(ph - prev) % 360:+.1f} deg" if prev is not None else ""))
    prev = ph
out(f"    omega = {np.degrees(OMEGA):.2f} deg.  A larger phase = an older (earlier) max, so z' = z exp(+i k omega) should move the answer to p0 - k.")

# ------------------------------------------------------------------ 3
x1, pM, pS, M, S, nb1 = gather_I1(500_000, 5920)
out(f"\n[3] I1 pool: seeds 5920..{5920 + nb1 - 1}, n = {len(x1)}.  Comparator transplant at t0 = pM + k <= 7.")
out("    donor: another selected sequence at ITS step pM' + k'; its comparator values are multiplied by M / M'.")
out("    rows: age k of the recipient's max (1..6);  entries P(ans = pM) | P(runner-up) | P(ans = t0)")
H1 = run(x1)
r1 = np.arange(len(x1))
pools = {}
for k in range(1, 8):
    v = np.flatnonzero(pM + k <= 7)
    pools[k] = (v, H1[v, (pM + k)[v]][:, COMP] / M[v, None])        # comparator state in units of the donor's max


def transplant(donor_comp):
    def fn(h, rows):
        h = h.copy()
        h[:, COMP] = donor_comp[rows]
        return h
    return fn


for k in range(1, 7):
    v, own = pools[k]
    t0 = (pM + k)[v]
    row = []
    for kd in (k, 1, 2, 3, 4, 5, 6):
        dv, dcomp = pools[kd]
        pick = rng.integers(0, len(dv), len(v))
        a = answer(run_iv(x1[v], t0, transplant(dcomp[pick] * M[v, None])))
        row.append(f"k'={kd}: {np.mean(a == pM[v]):.3f}|{np.mean(a == pS[v]):.3f}|{np.mean(a == t0):.3f}")
    out(f"    k={k}: same age -> " + "   ".join(row))
out("    same-age donors matched on the current input (|x_t0/M - x'_t0'/M'| smallest among 64 random candidates) and on the gap ((M-S)/M):")
for k in range(1, 7):
    v, own = pools[k]
    t0 = (pM + k)[v]
    xr = x1[v, t0] / M[v]; gr = (M[v] - S[v]) / M[v]
    cand = rng.integers(0, len(v), (len(v), 64))
    cost = np.abs(xr[cand] - xr[:, None]) + np.abs(gr[cand] - gr[:, None])
    cost[cand == np.arange(len(v))[:, None]] = np.inf
    pick = cand[np.arange(len(v)), cost.argmin(1)]
    a = answer(run_iv(x1[v], t0, transplant(own[pick] * M[v, None])))
    out(f"    k={k}: {np.mean(a == pM[v]):.3f}|{np.mean(a == pS[v]):.3f}|{np.mean(a == t0):.3f}   (median match error {np.median(cost.min(1)):.3f})")
