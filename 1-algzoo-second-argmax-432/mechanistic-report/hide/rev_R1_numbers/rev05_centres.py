"""Reviewer R1: (1) how much the orbit centre of each ordinary unit moves per unit of S (section 8: 'lowers the centre of
units 0, 10, 11, 12, 14, 15 by 6-10 per unit of S');  (2) e18's radius-x0 result split by the selection conditions.
Run:  python -B rev05_centres.py > rev05_centres_out.txt"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, LAM, coords
np.set_printoptions(precision=2, suppress=True, linewidth=220)
K = w[4]

x = sample(1_200_000, 5151)
H, Z = run(x, return_pre=True)
st = running_stats(x)
M, S, a1 = st["M1"], st["M2"], st["a1"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
r = Sp / np.where(M > 0, M, np.nan)
pre = np.zeros_like(H); pre[:, 1:] = H[:, :-1] @ W.T          # W h_(t-1): the pre-input state (as arm D used)
print("(1) orbit centre per unit M = mean over ages a = 2..9 of the mean of (W h_(t-1))_u / M, for three bins of S+/M (age = age of the max in h_(t-1) + 1)")
bins = [(-1, 0.05), (0.45, 0.55), (0.85, 0.95)]
cent = np.zeros((3, 16))
Mp, rp, ap = M[:, :-1], r[:, :-1], a1[:, :-1] + 1               # state before the input at t
for b, (lo, hi) in enumerate(bins):
    per_age = []
    for a in range(2, 10):
        q = (Mp > 0.3) & (ap == a) & (rp > lo) & (rp <= hi)
        per_age.append((pre[:, 1:][q] / Mp[q][:, None]).mean(0))
    cent[b] = np.mean(per_age, 0)
print("   unit:        " + "".join(f"{u:7d}" for u in ORD))
for b, (lo, hi) in enumerate(bins):
    print(f"   S/M in ({lo:5.2f},{hi:5.2f}]: " + "".join(f"{cent[b, u]:7.2f}" for u in ORD))
print("   change per unit S/M (bin 3 - bin 1)/0.9: " + "".join(f"{(cent[2, u] - cent[0, u]) / 0.9:7.2f}" for u in ORD))
print("   runner eigenvector x 10.157 x 5.2:       " + "".join(f"{v:7.2f}" for v in RIGHT['runner'].real * K * 5.2))

print("\n(2) e18 (clock radius x0 one step before the overtaking): P(answer = old max) under e18's selection and with conditions relaxed")
x = sample(1_500_000, 1818)
st = running_stats(x); truth = second_argmax(x); B = len(x); ar = np.arange(B)
tr = x.argmax(1)
tm1 = np.maximum(tr - 1, 0)
p0 = st["p1"][ar, tm1]; q0 = st["p2"][ar, tm1]; M0 = st["M1"][ar, tm1]; S0 = st["M2"][ar, tm1]
g = tr - 1 - p0


def run_mod(x, tr, f):
    Bn = len(x); h = np.zeros((Bn, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        sel = np.where(tr - 1 == t)[0]
        if len(sel) and f is not None:
            z = h[sel][:, ORD] @ LEFT["clock"]
            h[np.ix_(sel, ORD)] += 2 * np.real(np.multiply.outer((f - 1) * z, RIGHT["clock"]))
    return (h @ U.T).argmax(1)


base_sel = (tr >= 2) & (truth == p0) & (M0 > 0)
variants = {
    "e18: g>=2, S0>0.2, M0-S0>0.1, third=q0, 3<=tr<=8": base_sel & (tr >= 3) & (g >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (st["p3"][:, 9] == q0) & (tr <= 8),
    "g>=2, S0>0, any gap, tr<=8                      ": base_sel & (g >= 2) & (S0 > 0) & (tr <= 8),
    "g>=2, S0>0, any gap, tr<=9                      ": base_sel & (g >= 2) & (S0 > 0),
    "g>=1, S0>0, any gap, tr<=9 (arm F: age>=2 at the record)": base_sel & (g >= 1) & (S0 > 0),
    "g==1 only, S0>0                                 ": base_sel & (g == 1) & (S0 > 0),
}
for lab, ok in variants.items():
    b0 = run_mod(x[ok], tr[ok], None); a0 = run_mod(x[ok], tr[ok], 0.0); a5 = run_mod(x[ok], tr[ok], 0.5); a2 = run_mod(x[ok], tr[ok], 2.0)
    print(f"   {lab}: n={ok.sum():7d}  unmodified {np.mean(b0 == p0[ok]):.3f}   x0: {np.mean(a0 == p0[ok]):.3f}   x0.5: {np.mean(a5 == p0[ok]):.3f}   x2: {np.mean(a2 == p0[ok]):.3f}")
