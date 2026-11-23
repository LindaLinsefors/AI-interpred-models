"""E33. What the tape modes' piece of the scores, eq. (4a) of tape.md, holds as a function of the inputs.
    G[i] = zeta_alt p_alt[i] + sum_{short, h2, h3} 2 Re(z_m p_m[i]),   s = G + R  (R: all other modes + pulse units)
Final states (t = 9; by the zero-prefix identity only ages matter), entries of age >= 2 (indices 0..7),
maximum and runner-up both at least two steps old, M > 0.3, S > 0.1.  Everything in units of M.
Run:  python -B hide/e33_tape_piece.py > hide/outputs/e33_tape_piece.txt"""
from common import *
from modes import RIGHT, coords

B = 1_000_000
x = sample(B, 3333)
h = run(x)[:, 9]
c = coords(h)
Uo = U[:, ORD]
G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
for n in ("short", "h2", "h3"):
    G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
s = h @ U.T
R = s - G
order = np.argsort(-x, 1)
pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
ar = np.arange(B)
M, S, T3 = x[ar, pM], x[ar, pS], x[ar, pT]
ok = (pM <= 7) & (pS <= 7) & (M > 0.3) & (S > 0.1)
print(f"{ok.sum()} of {B} final states selected (max and runner-up at least two steps old, M > 0.3, S > 0.1)")
med = lambda v: float(np.median(v)) if len(v) else float("nan")
rel = lambda A: (A - A[ar, pS][:, None]) / M[:, None]            # entry minus the runner-up's entry, in units of M
g, f, r = rel(G), rel(s), rel(R)
idx = np.arange(8)[None, :]
v = (x[:, :8] - S[:, None]) / M[:, None]
below = ok[:, None] & (x[:, :8] < S[:, None])
ismax = ok[:, None] & (idx == pM[:, None])

print("\n1. Which entry is the largest among indices 0..7?          runner-up   max    third   other")
for name, A in (("tape piece G", G), ("rest R", R), ("full score s", s)):
    top = A[:, :8].argmax(1)
    print(f"   {name:13s}                                         {np.mean(top[ok] == pS[ok]):.3f}    {np.mean(top[ok] == pM[ok]):.3f}  {np.mean(top[ok] == pT[ok]):.3f}   "
          f"{np.mean((top[ok] != pS[ok]) & (top[ok] != pM[ok]) & (top[ok] != pT[ok])):.3f}")

print("\n2. Inputs below the runner-up: median of (entry - runner-up's entry)/M by v = (x - S)/M")
edges = [-9, -1.5, -0.8, -0.4, -0.2, -0.1, -0.05, -0.02, 0]
print("   v bin             n        tape G     rest R    full s  | IQR of G")
for lo, hi in zip(edges[:-1], edges[1:]):
    q = below & (v > lo) & (v <= hi)
    print(f"   ({lo:5.2f},{hi:5.2f}]  {q.sum():8d}   {med(g[:, :8][q]):8.1f}  {med(r[:, :8][q]):8.1f}  {med(f[:, :8][q]):8.1f}  | "
          f"{np.percentile(g[:, :8][q], 25):6.1f} to {np.percentile(g[:, :8][q], 75):6.1f}")
q = below & (v > -0.1)
for name, A in (("G", g), ("R", r), ("s", f)):
    X = np.stack([v[q], np.ones(q.sum())], 1); b = np.linalg.lstsq(X, A[:, :8][q], rcond=None)[0]
    print(f"   slope near the runner-up (-0.1 < v < 0), {name}: {b[0]:7.1f} per unit of (x - S), intercept {b[1]:6.1f} M")

print("\n3. Floor (v < -0.4) by order of arrival relative to the top two, and by age of the entry: median (entry - runner-up's)/M")
lo_, hi_ = np.minimum(pM, pS)[:, None], np.maximum(pM, pS)[:, None]
far = below & (v < -0.4)
for name, cond in (("after both", idx > hi_), ("between them", (idx > lo_) & (idx < hi_)), ("before both", idx < lo_)):
    q = far & cond
    print(f"   {name:13s} n = {q.sum():8d}   G {med(g[:, :8][q]):7.1f}   R {med(r[:, :8][q]):7.1f}   s {med(f[:, :8][q]):7.1f}")
print("   by age of the entry 2..9:  G", " ".join(f"{med(g[:, :8][far & (idx == 9 - a)]):6.1f}" for a in range(2, 10)))
print("                              s", " ".join(f"{med(f[:, :8][far & (idx == 9 - a)]):6.1f}" for a in range(2, 10)))

print("\n4. The maximum's entry minus the runner-up's entry, by age of the maximum: medians, and fit  a M + b (M - S)")
print("   age    n       G      R      s   |  G: a      b      R2  |  R: a      b      R2  |  s: a      b")
for a in range(2, 10):
    q = ok & (pM == 9 - a)
    Xf = np.stack([M[q], (M - S)[q]], 1)
    out = []
    for A in (G, R, s):
        y = A[ar, pM][q] - A[ar, pS][q]
        b = np.linalg.lstsq(Xf, y, rcond=None)[0]
        out.append((med(y / M[q]), b, 1 - ((y - Xf @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    print(f"   {a}  {q.sum():7d} {out[0][0]:6.1f} {out[1][0]:6.1f} {out[2][0]:6.1f}  | {out[0][1][0]:6.1f} {out[0][1][1]:6.1f}  {out[0][2]:5.2f}  | "
          f"{out[1][1][0]:6.1f} {out[1][1][1]:6.1f}  {out[1][2]:5.2f}  | {out[2][1][0]:6.1f} {out[2][1][1]:6.1f}")
D = (pS - pM)
print("   the same G difference by signed distance pS - pM (median, gap coefficient b):")
for d in range(-7, 8):
    q = ok & (D == d)
    if q.sum() > 3000:
        Xf = np.stack([M[q], (M - S)[q]], 1); y = G[ar, pM][q] - G[ar, pS][q]
        b = np.linalg.lstsq(Xf, y, rcond=None)[0]
        print(f"     d = {d:+d}: median {med(y / M[q]):6.1f} M,  a = {b[0]:6.1f},  b = {b[1]:6.1f}")

print("\n5. How much of the full relative score is the tape piece? (inputs below the runner-up, ages >= 2)")
gg, ff, rr = g[:, :8][below], f[:, :8][below], r[:, :8][below]
print(f"   corr(G, s) = {np.corrcoef(gg, ff)[0, 1]:.3f},  corr(R, s) = {np.corrcoef(rr, ff)[0, 1]:.3f},  corr(G, R) = {np.corrcoef(gg, rr)[0, 1]:.3f};  "
      f"rms G {np.sqrt((gg ** 2).mean()):.1f}, R {np.sqrt((rr ** 2).mean()):.1f}, s {np.sqrt((ff ** 2).mean()):.1f}")

print("\n6. The runner-up's own entry of G, absolute: fit G[pS] = a M + b S, by age of the runner-up, split by arrival order")
for name, cond in (("runner-up arrived after the max", pS > pM), ("runner-up arrived before the max", pS < pM)):
    print("   " + name)
    for a in range(2, 10):
        q = ok & cond & (pS == 9 - a)
        if q.sum() > 3000:
            Xf = np.stack([M[q], S[q]], 1); y = G[ar, pS][q]
            b = np.linalg.lstsq(Xf, y, rcond=None)[0]
            print(f"     age {a}: n = {q.sum():6d}  a = {b[0]:6.1f}  b = {b[1]:6.1f}  R2 = {1 - ((y - Xf @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum():.2f}   "
                  f"(mean of G over indices 0..7: {med(G[q][:, :8].mean(1) / M[q]):6.1f} M)")
