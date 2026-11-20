"""Reviewer R1: how good is the one-line law  V[p] ~ -60 (M - 0.87 S) cos(42.75 (p - p_M))  as a replacement for the
comparators' vote?  t = 9, all four comparators on, S > 0.1, max and runner-up both at positions 0..7 (so only indices
where the law is claimed).  Scores restricted to positions 0..7:  s = m + V  versus  s' = m + V_formula.
Run:  python -B rev07_formula_as_V.py > rev07_formula_as_V_out.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
om = np.radians(42.748)
x = sample(1_000_000, 3131)
H, Z = run(x, return_pre=True)
st = running_stats(x)
t = 9
V = H[:, t][:, COMP] @ U[:, COMP].T
s = H[:, t] @ U.T
m = s - V
M, S, pM, pS = st["M1"][:, t], st["M2"][:, t], st["p1"][:, t], st["p2"][:, t]
ok = (Z[:, t][:, COMP] > 0).all(1) & (S > 0.1) & (pM <= 7) & (pS <= 7)
p = np.arange(8)[None, :]
cosd = np.cos(om * (p - pM[:, None]))
print(f"n = {ok.sum()} of {len(x)} sequences (t = 9, all four comparators on, S > 0.1, max and runner-up at positions 0-7); argmax over positions 0-7")
def rate(score):
    a = score[ok][:, :8].argmax(1)
    return np.mean(a == pS[ok]), np.mean(a == pM[ok])
r = rate(s); print(f"  true scores s = m + V:                         runner-up first {r[0]:.3f}, max first {r[1]:.3f}")
r = rate(m); print(f"  m alone:                                       runner-up first {r[0]:.3f}, max first {r[1]:.3f}")
for A0, k in ((60, 0.87), (66.2, 0.891)):
    Vf = -A0 * (M - k * S)[:, None] * cosd
    r = rate(m[:, :8] + Vf)
    vv = V[ok][:, :8]; ff = Vf[ok]
    r2c = 1 - ((vv - ff) ** 2).sum() / ((vv - vv.mean(1, keepdims=True)) ** 2).sum()
    print(f"  m + [-{A0} (M - {k} S) cos(42.75 (p - p_M))]:   runner-up first {r[0]:.3f}, max first {r[1]:.3f};   rms error of the law {np.sqrt(((vv - ff) ** 2).mean()):.1f} logits (rms of V centred over positions {np.sqrt(((vv - vv.mean(1, keepdims=True)) ** 2).mean()):.1f}; median margin of the runner-up in s: {np.median(np.sort(s[ok][:, :8], 1)[:, -1] - np.sort(s[ok][:, :8], 1)[:, -2]):.1f})")
# per-age regression R2 restricted to indices 0..7 (the report quotes 0.88-0.97, which pools all ten indices)
a1 = st["a1"][:, t]
allon = (Z[:, t][:, COMP] > 0).all(1) & (S > 0.1)
out10, out8 = [], []
for a in range(1, 10):
    q = allon & (a1 == a)
    X = np.column_stack([(M - S)[q], S[q], x[q, t]])
    co, *_ = np.linalg.lstsq(X, V[q], rcond=None)
    Y = V[q]; res = Y - X @ co
    out10.append(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    out8.append(1 - (res[:, :8] ** 2).sum() / ((Y[:, :8] - Y[:, :8].mean(0)) ** 2).sum())
print("  R2 of the per-age regression V ~ A (M-S) + B S + C x_9, ages 1..9: all ten indices", np.round(out10, 3), "  indices 0-7 only", np.round(out8, 3))
