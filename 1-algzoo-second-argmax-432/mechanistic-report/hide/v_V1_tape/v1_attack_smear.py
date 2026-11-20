"""Verifier V1: neighbour coupling of tape entries, robustness (follow-up of E4 / E4b).
Overwritten input p arrived AFTER both top-2 inputs (no event at any v' < 0), has age >= 2 at t = 9 and both
neighbours p-1, p+1 are neither max nor runner-up.  x_p' = S + v' M for several v'.  Seed 6401 (+6402), 300k sequences.
Reported per pair (v'_a -> v'_b): median change of the own entry, of the two neighbours, of entries at distance >= 2,
and the ratio neighbour / own.   Also a mirror test: does the own entry respond to a NEIGHBOUR's value?
Run: python -B v1_attack_smear.py -> v1_attack_smear.txt
"""
from vlib import *

P = Tee("v1_attack_smear.txt")
x = sample(300_000, 6401)
rng = np.random.default_rng(6402)
p3, val = top3(x, 9)
p1, p2, M, S = p3[:, 0], p3[:, 1], val[:, 0], val[:, 1]
top = np.maximum(p1, p2)
keep = (S > 0.2) & (top <= 4)                         # positions top+2 .. 7 are candidates (neighbours top+1.., p+1 <= 8)
x, p1, p2, M, S, top = x[keep], p1[keep], p2[keep], M[keep], S[keep], top[keep]
n = len(x); r = np.arange(n)
lo = top + 2
p = (lo + np.floor(rng.random(n) * (8 - lo))).astype(int)      # uniform in top+2 .. 7  -> age 2..(7-top)
assert (p >= top + 2).all() and (p <= 7).all()
grid = [-2.5, -1.2, -0.6, -0.3, -0.15, -0.075]
D = {}
for vv in grid:
    xm = x.copy(); xm[r, p] = S + vv * M
    s = run(xm)[:, 9] @ U.T
    D[vv] = (s - s[r, p2][:, None]) / M[:, None]
far = np.ones((n, 10), bool)
far[r, p] = False; far[r, p - 1] = False; far[r, p + 1] = False; far[r, p1] = False; far[r, p2] = False
P(f"sequences: {n}; overwritten input of age {int((9 - p).min())}..{int((9 - p).max())} at t=9, arrived after both top-2")
P(" pair (v'a -> v'b) | own entry: median change | neighbour p-1 | neighbour p+1 | distance>=2 (median, median|.|) | the max's entry | ratio mean(neighbours)/own")
for a, b in ((-2.5, -0.075), (-2.5, -0.6), (-0.6, -0.075), (-0.3, -0.075), (-1.2, -0.3), (-0.15, -0.075)):
    d = D[b] - D[a]
    own = np.median(d[r, p]); nm = np.median(d[r, p - 1]); npl = np.median(d[r, p + 1])
    P(f" {a:6.3f} -> {b:6.3f} | {own:7.2f} | {nm:7.2f} | {npl:7.2f} | {np.median(d[far]):6.2f}, {np.median(np.abs(d[far])):5.2f} | {np.median(d[r, p1]):6.2f} | {(nm + npl) / 2 / own:5.2f}")
P(" per-sequence ratio (neighbour change / own change) for -0.6 -> -0.075: median p-1: %.2f, p+1: %.2f" % (
    np.median((D[-0.075] - D[-0.6])[r, p - 1] / (D[-0.075] - D[-0.6])[r, p]), np.median((D[-0.075] - D[-0.6])[r, p + 1] / (D[-0.075] - D[-0.6])[r, p])))
P(" own entry level, median d_own by v': " + "  ".join(f"{vv}: {np.median(D[vv][r, p]):.2f}" for vv in grid))
P(" neighbour p+1 level (its own value untouched), median d by v' of p: " + "  ".join(f"{vv}: {np.median(D[vv][r, p + 1]):.2f}" for vv in grid))
P(" neighbour p-1 level, median d by v' of p: " + "  ".join(f"{vv}: {np.median(D[vv][r, p - 1]):.2f}" for vv in grid))
# does a neighbour ever overtake the runner-up because of p's value?
for vv in (-0.6, -0.075):
    P(f" v' = {vv}: P(network answer = p2) {np.mean(D[vv].argmax(1) == p2):.4f};  P(answer = p) {np.mean(D[vv].argmax(1) == p):.4f};  P(answer = p-1 or p+1) {np.mean(np.abs(D[vv].argmax(1) - p) == 1):.4f}")
