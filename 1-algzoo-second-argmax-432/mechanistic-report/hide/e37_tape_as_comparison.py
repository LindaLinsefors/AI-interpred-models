"""E37. How good is "the tape entry is a clipped comparison of its input with the runner-up"?
D[i] = entry i minus the runner-up's entry, for the tape piece (eq. 4a of tape.md) and for the full score.
Judged the way the network uses it: is the sign right, and how accurate is the comparison near the runner-up
(expressed as an error in the input, in units of M)?  Same selection as e33-e35.
Run:  python -B hide/e37_tape_as_comparison.py > hide/outputs/e37_tape_as_comparison.txt"""
from common import *
from modes import RIGHT, coords

B = 1_000_000
x = sample(B, 3737)
h = run(x)[:, 9]; c = coords(h); Uo = U[:, ORD]
G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
for n in ("short", "h2", "h3"):
    G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
s = h @ U.T
order = np.argsort(-x, 1); pM, pS = order[:, 0], order[:, 1]; ar = np.arange(B)
M, S = x[ar, pM], x[ar, pS]
ok = (pM <= 7) & (pS <= 7) & (M > 0.3) & (S > 0.1)
idx = np.arange(8)[None, :]
below = ok[:, None] & (x[:, :8] < S[:, None])
u = (S[:, None] - x[:, :8]) / M[:, None]                              # depth below the runner-up, units of M
age = 9 - np.broadcast_to(idx, u.shape)
for name, A in (("tape piece", G), ("full score", s)):
    D = (A[:, :8] - A[ar, pS][:, None]) / M[:, None]
    print(f"\n===== {name}: D = (entry - runner-up's entry) / M, inputs below the runner-up, ages 2..9")
    print("  depth u = (S - x)/M      n     P(D < 0)   median D   10%..90% of D")
    for lo, hi in ((0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 9)):
        q = below & (u > lo) & (u <= hi); d = D[q]
        print(f"  {lo:4.2f}-{hi:4.2f}          {q.sum():8d}   {np.mean(d < 0):.3f}    {np.median(d):6.1f}    {np.percentile(d, 10):6.1f} .. {np.percentile(d, 90):6.1f}")
    q = below & (u < 0.1)
    X = np.stack([-u[q], np.ones(q.sum())], 1); b = np.linalg.lstsq(X, D[q], rcond=None)[0]; res = D[q] - X @ b
    print(f"  near the runner-up (u < 0.1): D = {b[0]:.1f} (x - S)/M {b[1]:+.2f};  R2 = {1 - res.var() / D[q].var():.3f};  residual: sd {res.std():.2f}, quartiles {np.percentile(res, 25):+.2f} / {np.percentile(res, 75):+.2f}")
    print(f"     as an error in the input: sd {res.std() / b[0]:.4f} M, quartiles {np.percentile(res, 25) / b[0]:+.4f} / {np.percentile(res, 75) / b[0]:+.4f} M")
    print("     the same fit by age of the entry:  slope, R2, input-error quartile half-width (M)")
    for a in range(2, 10):
        qa = q & (age == a)
        X = np.stack([-u[qa], np.ones(qa.sum())], 1); b = np.linalg.lstsq(X, D[qa], rcond=None)[0]; res = D[qa] - X @ b
        print(f"       age {a}: {b[0]:6.1f}   {1 - res.var() / D[qa].var():.3f}   {(np.percentile(res, 75) - np.percentile(res, 25)) / 2 / b[0]:.4f}")
    dm = (A[ar, pM] - A[ar, pS])[ok] / M[ok]
    print(f"  the maximum's entry: P(D < 0) = {np.mean(dm < 0):.3f}, median {np.median(dm):.1f}, 10%..90% {np.percentile(dm, 10):.1f} .. {np.percentile(dm, 90):.1f}")
