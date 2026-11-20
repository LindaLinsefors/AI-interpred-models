"""I1: the comparator state as two components.  Seeds 6000, 6001.

  h[COMP] = (clock share) + (rest),   clock share = 2 Re(z RIGHT[clock]) on the comparators.
  Intervention at t0 = pM + k <= 7:  h[COMP] := c_rest * rest + c_clock * clock share  (clipped at 0).
  (1, 1) = nothing; (0, 0) = the lead's wipe; (1, 0) = remove only the clock share; (0, 1) = keep only the clock share.
Also: logit margins after the wipe (is 'answers the max' a near-tie with the input of the wipe step?).
"""
from v3lib import *

out = Tee("a9_two_components.txt")
isC = np.isin(ORD_A, COMP)
x, pM, pS, M, S, nb = gather_I1(500_000, 6000)
out(f"I1 pool: seeds 6000..{6000 + nb - 1}, n = {len(x)}")


def two(c_rest, c_clock, clip=True):
    def fn(h, rows):
        z = h[:, ORD_A] @ LEFT["clock"]
        d = 2 * np.real(np.multiply.outer(z, RIGHT["clock"]))[:, isC]
        h = h.copy()
        h[:, COMP] = c_rest * (h[:, COMP] - d) + c_clock * d
        return np.maximum(h, 0.0) if clip else h
    return fn


out("\nP(ans = pM) | P(ans = true runner-up) | P(ans = t0), for k = 1..6 (t0 <= 7)")
for c_rest in (1, 0.9, 0.5, 0.1, 0.02, 0):
    for c_clock in (1, 0.75, 0.5, 0.25, 0, -0.5):
        row = []
        for k in range(1, 7):
            v = pM + k <= 7
            t0 = (pM + k)[v]
            a = answer(run_iv(x[v], t0, two(c_rest, c_clock)))
            row.append(f"{np.mean(a == pM[v]):.3f}|{np.mean(a == pS[v]):.3f}|{np.mean(a == t0):.3f}")
        out(f"  rest x {c_rest:<4} clock share x {c_clock:<5}: " + "  ".join(row))

out("\nlogit margins at t = 9 after the lead's wipe (units of M), k = 1..6, t0 <= 7:")
out("  median and 10th percentile of s[pM] - s[t0];  median of s[pM] - (largest other logit);  and the same margins when only the clock share is removed")
for k in range(1, 7):
    v = pM + k <= 7
    t0 = (pM + k)[v]
    rr = np.arange(v.sum())
    res = []
    for fn in (scale_units(COMP, 0.0), two(1, 0)):
        s = run_iv(x[v], t0, fn) @ U.T
        m1 = (s[rr, pM[v]] - s[rr, t0]) / M[v]
        so = s.copy(); so[rr, pM[v]] = -np.inf
        m2 = (s[rr, pM[v]] - so.max(1)) / M[v]
        res.append(f"{np.median(m1):+7.2f} {np.percentile(m1, 10):+7.2f} {np.median(m2):+7.2f}")
    s0 = run(x[v])[:, -1] @ U.T
    base = (s0[rr, pS[v]] - s0[rr, pM[v]]) / M[v]
    out(f"  k={k}: wipe: {res[0]}   | clock share removed: {res[1]}   | unperturbed median s[runner-up] - s[pM] = {np.median(base):+.2f}")
