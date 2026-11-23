"""D2b. Why two inputs below the runner-up are not additive: the write of the second, with and without the first.
M = 1 at position 0, S at position 1 (or the reverse), first input v1 = f1 S at position p1, second input v2 swept at
position p2 > p1, filler 0.  Contribution of the second = tape(all) - tape(without the second), read at its age 2.
A = projection on the pattern kA (d1), Theta = intercept of the straight line A = g (v2 - Theta) above onset.
Run:  python -B d2b_second_write.py > d2b_second_write.txt"""
from dict_common import *

kA = np.load(HERE / "d1_patterns.npz")["kA"]
VS = np.round(np.arange(0.0, 1.0, 0.005), 3)


def second(pM, pS, S, p1, v1, p2, fill=0.0):
    """g, Theta, resid of the second input's write, and the size of the change of the FIRST input's contribution."""
    X = np.full((len(VS), 10), fill); X[:, pM], X[:, pS] = 1.0, S
    if v1 is not None: X[:, p1] = v1
    Xr = X[:1].copy(); X[:, p2] = VS
    H, Hr = run(X), run(Xr)
    D = to7(zc(H[:, p2 + 2]) - zc(Hr[:, p2 + 2])); A = D @ kA / (kA @ kA); u = H[:, p2 + 1, 5] - Hr[:, p2 + 1, 5]
    lo = (VS < S) & (u > 0.05)
    if lo.sum() < 3: return np.nan, np.nan, np.nan, A
    b = np.polyfit(VS[lo], A[lo], 1)
    return b[0], -b[1] / b[0], np.median(np.linalg.norm(D[lo] - np.outer(A[lo], kA), axis=1) / np.linalg.norm(D[lo], axis=1)), A


print("1. Threshold Theta_2 / S of the second input (slope g in brackets) by the value of the first, f1 = v1 / S.  S = 0.6, filler 0.")
print("   'none' = no first input.  Rows: positions (pM, pS, p1, p2).")
F1 = (None, 0.5, 0.8, 0.9, 0.95, 0.99)
print("                        " + "  ".join(f"{'none' if f is None else f:>12}" for f in F1))
rows = []
for pM, pS in ((0, 1), (1, 0)):
    for p1, p2 in ((2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 6), (4, 5), (4, 7), (5, 6), (5, 7)):
        r = [second(pM, pS, 0.6, p1, None if f is None else f * 0.6, p2) for f in F1]
        rows.append([x[1] / 0.6 for x in r])
        print(f"   ({pM}, {pS}, {p1}, {p2})        " + "  ".join(f"{x[1] / 0.6:5.2f} ({x[0]:4.1f})" for x in r))
rows = np.array(rows)
print("   median over rows       " + "  ".join(f"{np.nanmedian(rows[:, j]):12.2f}" for j in range(len(F1))))

print("\n2. Is the raised threshold 'the first input itself'?  Theta_2 against v1, both in units of S, all cases of section 1 plus S = 0.3, 0.9")
allr = []
for S in (0.3, 0.6, 0.9):
    for pM, pS in ((0, 1), (1, 0)):
        for p1 in range(2, 7):
            for p2 in range(p1 + 1, 8):
                th0 = second(pM, pS, S, p1, None, p2)[1] / S
                for f in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
                    g, th, rs, _ = second(pM, pS, S, p1, f * S, p2)
                    allr.append((S, p2 - p1, f, th0, th / S, g))
allr = np.array(allr)
print("   f1 = v1/S | Theta_2/S median (10%..90%) | Theta without first: median | Theta_2 - max(Theta_0, v1) in units of S: median (10%..90%) | slope g median")
for f in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
    s = allr[:, 2] == f; t2, t0 = allr[s, 4], allr[s, 3]; d = t2 - np.maximum(t0, f)
    print(f"     {f:4.2f}    |  {np.nanmedian(t2):.3f} ({np.nanquantile(t2, .1):.3f}..{np.nanquantile(t2, .9):.3f})   |   {np.nanmedian(t0):.3f}   |"
          f"   {np.nanmedian(d):+.3f} ({np.nanquantile(d, .1):+.3f}..{np.nanquantile(d, .9):+.3f})   |  {np.nanmedian(allr[s, 5]):5.1f}")
print("   by the gap p2 - p1, for f1 = 0.9:  Theta_2/S median: " + "  ".join(
    f"gap {k}: {np.nanmedian(allr[(allr[:, 2] == 0.9) & (allr[:, 1] == k), 4]):.2f}" for k in range(1, 6)))

print("\n3. Law of the raised threshold:  Theta_2 = Theta_0 + beta (v1 - Theta_0)_+  (all in units of S), cases of section 2 with finite fits")
ok = np.isfinite(allr[:, 3]) & np.isfinite(allr[:, 4])
e = np.maximum(allr[ok, 2] - allr[ok, 3], 0); dth = allr[ok, 4] - allr[ok, 3]; s = e > 0.02
beta = (e[s] @ dth[s]) / (e[s] @ e[s])
print(f"   n = {s.sum()} with v1 more than 0.02 S above Theta_0:  beta = {beta:.3f},  R2 = {1 - ((dth[s] - beta * e[s]) ** 2).sum() / ((dth[s] - dth[s].mean()) ** 2).sum():.3f},"
      f"  ratio (Theta_2 - Theta_0) / (v1 - Theta_0): median {np.median(dth[s] / e[s]):.3f}, 10%..90% {np.quantile(dth[s] / e[s], .1):.3f}..{np.quantile(dth[s] / e[s], .9):.3f}")
print(f"   cases with v1 below Theta_0 (n = {(~s).sum()}): Theta_2 - Theta_0 in units of S: median {np.median(dth[~s]):+.3f}, 10%..90% {np.quantile(dth[~s], .1):+.3f}..{np.quantile(dth[~s], .9):+.3f}")
print(f"   cases without a finite fit (unit 5 fires on fewer than 3 grid values below S): {(~ok).sum()} of {len(ok)}")
