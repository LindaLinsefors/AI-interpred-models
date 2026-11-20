"""Reviewer R1: where do '60', '8' and '0.87' in  V[p] ~ -60 (M - 0.87 S) cos(42.75 (p - p_M))  come from?
Same sample and selection as e15 (seed 1515, t = 9, all four comparators on, S > 0.1).  Least-squares versions.
Run:  python -B rev06_vote_fit.py > rev06_vote_fit_out.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
om = np.radians(42.748)
x = sample(500_000, 1515)
H, Z = run(x, return_pre=True)
st = running_stats(x)
t = 9
V = H[:, t][:, COMP] @ U[:, COMP].T
allon = (Z[:, t][:, COMP] > 0).all(1) & (st["M2"][:, t] > 0.1)
Mx, Sx, a1 = st["M1"][:, t], st["M2"][:, t], st["a1"][:, t]
A = np.full((10, 10), np.nan); Bt = np.full((10, 10), np.nan)
for a in range(1, 10):
    s = allon & (a1 == a)
    X = np.column_stack([(Mx - Sx)[s], Sx[s], x[s, t]])
    co, *_ = np.linalg.lstsq(X, V[s], rcond=None)
    A[a], Bt[a] = co[0], co[1]
d, Av, Bv = [], [], []
for a in range(2, 10):
    for idx in range(8):
        d.append(idx - (9 - a)); Av.append(A[a, idx]); Bv.append(Bt[a, idx])
d, Av, Bv = np.array(d), np.array(Av), np.array(Bv)
c = np.cos(om * d)
X = np.column_stack([c, np.ones_like(c)])
ca, *_ = np.linalg.lstsq(X, Av, rcond=None)
cb, *_ = np.linalg.lstsq(X, Bv, rcond=None)
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"A(d) ~ {ca[0]:.1f} cos(42.75 d) + {ca[1]:.1f}   R2 {r2(Av, X @ ca):.3f}      (report: -60 cos)")
print(f"B(d) ~ {cb[0]:.1f} cos(42.75 d) + {cb[1]:.1f}   R2 {r2(Bv, X @ cb):.3f}      (report: 'same shape with an amplitude of about 8')")
print(f"=> V ~ {ca[0]:.1f} [ (M - S) + {cb[0]/ca[0]:.3f} S ] cos = {ca[0]:.1f} (M - {1 - cb[0]/ca[0]:.2f} S) cos   plus offsets {ca[1]:.1f} (M - S) + {cb[1]:.1f} S")
for lab, sel in (("d >= 0", d >= 0), ("d <= 0", d <= 0)):
    a_ = (Av[sel] * c[sel]).sum() / (c[sel] ** 2).sum(); b_ = ((Bv[sel] - Bv[sel].mean()) * c[sel]).sum() / (c[sel] ** 2).sum()
    print(f"   {lab}: amplitude of A {a_:.1f}")
# direct fit on states: V[idx] for idx <= 7 against -(M - k S) cos(om (idx - idx_M)), ages 2..9, one amplitude and one k
rows_c, rows_V, rows_M, rows_S = [], [], [], []
for a in range(2, 10):
    s = allon & (a1 == a)
    for idx in range(8):
        cc = np.cos(om * (idx - (9 - a)))
        rows_c.append(np.full(s.sum(), cc)); rows_V.append(V[s, idx]); rows_M.append(Mx[s]); rows_S.append(Sx[s])
cc = np.concatenate(rows_c); VV = np.concatenate(rows_V); MM = np.concatenate(rows_M); SS = np.concatenate(rows_S)
X = np.column_stack([MM * cc, SS * cc])
co, *_ = np.linalg.lstsq(X, VV, rcond=None)
print(f"direct state-level fit, idx 0..7, ages 2..9:  V ~ {co[0]:.1f} (M - {-co[1]/co[0]:.3f} S) cos(42.75 (p - p_M)),  R2 (uncentred) {1 - ((VV - X @ co) ** 2).sum() / (VV ** 2).sum():.3f}")
p60 = -60 * (MM - 0.87 * SS) * cc
print(f"   the report's formula -60 (M - 0.87 S) cos: R2 (uncentred) {1 - ((VV - p60) ** 2).sum() / (VV ** 2).sum():.3f}")
# index 8
print("index 8: A(a1, 8) for a1 = 2..9:", np.round(A[2:, 8], 1), "  -60 cos(42.75 (a1 - 1)):", np.round(-60 * np.cos(om * (np.arange(2, 10) - 1)), 1))
rows = []
for a in range(2, 10):
    s = allon & (a1 == a)
    rows.append((V[s, 8], -60 * (Mx[s] - 0.87 * Sx[s]) * np.cos(om * (a - 1))))
v8 = np.concatenate([r_[0] for r_ in rows]); p8 = np.concatenate([r_[1] for r_ in rows])
print(f"   state-level R2 of the formula at index 8 (uncentred): {1 - ((v8 - p8) ** 2).sum() / (v8 ** 2).sum():.3f};  correlation {np.corrcoef(v8, p8)[0, 1]:.3f}")
print("index 9: mean over a1 = 1..9 of A, B:", round(float(np.nanmean(A[1:, 9])), 1), round(float(np.nanmean(Bt[1:, 9])), 1),
      " => V[9] ~", round(float(np.nanmean(A[1:, 9])), 1), "M +", round(float(np.nanmean(Bt[1:, 9]) - np.nanmean(A[1:, 9])), 1), "S + 58.2 x_9;  A by age:", np.round(A[1:, 9], 1))
