"""E05. Cell-wise 'two-value' description of the state in mode coordinates.

For each cell (a1, a2) = (age of max, age of runner-up), steps with S > 0.05:
    m_k(t) / 10.157  ~  alpha_k(a1, a2) * (M - S)  +  beta_k(a1, a2) * S
(complex least squares, no intercept).  If the state is a superposition of a rotating
'increment record' of the max and a rotating record of the runner-up, then alpha_k should
depend on a1 only and beta_k mainly on a2.
"""
from common import *
from modes import coords, NAMES, LAM, IS_COMPLEX

x = sample(1_000_000, 13)
H = run(x)
st = running_stats(x)
m = coords(H)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
ok = np.isfinite(S) & (S > 0.05)

res = {}
for A1 in range(10):
    for A2 in range(10):
        if A1 == A2:
            continue
        s = ok & (a1 == A1) & (a2 == A2)
        if s.sum() < 2000:
            continue
        X = np.column_stack([M[s] - S[s], S[s]])
        for n in NAMES:
            y = m[n][s] / 10.157
            b, *_ = np.linalg.lstsq(X.astype(complex), y.astype(complex), rcond=None)
            r = y - X @ b
            R2 = 1 - (np.abs(r) ** 2).sum() / (np.abs(y) ** 2).sum()
            res[(A1, A2, n)] = (b[0], b[1], R2, s.sum())

def fmt(c, cplx):
    return f"{abs(c):5.2f}@{np.degrees(np.angle(c)):5.0f}" if cplx else f"{c.real:6.2f}     "

for n in NAMES:
    cplx = IS_COMPLEX[n]
    print(f"\n=== mode {n}  (lam = {abs(LAM[n]):.3f} @ {np.degrees(np.angle(LAM[n])):.1f} deg) ===")
    for which, idx in (("alpha (coef of M-S)", 0), ("beta (coef of S)", 1), ("R2", 2)):
        print(f"  {which}: rows a1 = 0..9, columns a2 = 0..9")
        for A1 in range(10):
            row = ""
            for A2 in range(10):
                v = res.get((A1, A2, n))
                if v is None:
                    row += "     .     "
                elif idx == 2:
                    row += f"   {v[2]:5.3f}   "
                else:
                    row += fmt(v[idx], cplx)
            print(f"    a1={A1}: {row}")
import pickle
pickle.dump(res, open(HIDE / "e05_cell_traces.pkl", "wb"))
