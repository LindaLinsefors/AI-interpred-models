"""E16. What are the 'max scores' m (readout without comparators) as a function of the input values?
And how does the runner-up's margin over the max split into m-part and comparator vote V?
Final step t = 9 only, inputs older than the last one, all four comparators on at t = 9, S > 0.2."""
from common import *

x = sample(800_000, 1616)
H, Z = run(x, return_pre=True)
st = running_stats(x)
B = len(x); ar = np.arange(B); t = 9
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
s_all = H[:, t] @ U.T
v_all = H[:, t] @ Uc.T
m_all = s_all - v_all
M, S, T3 = st["M1"][:, t], st["M2"][:, t], st["M3"][:, t]     # three largest values (T3: not T, the sequence length from common)
pM, pS, pT = st["p1"][:, t], st["p2"][:, t], st["p3"][:, t]
ok = (Z[:, t][:, COMP] > 0).all(1) & (S > 0.2) & (pM < 9) & (pS < 9)

def med(a):
    return f"{np.median(a):7.1f} [{np.percentile(a,25):6.1f},{np.percentile(a,75):6.1f}]"

print("margins in units of M, by relative gap g = (M - S)/M and separation D = |pM - pS|  (median [IQR])")
print("  s-margin = s[pS] - s[pM];  m-part = m[pS] - m[pM];  V-part = V[pS] - V[pM];  prediction for V-part: 60 (M - 0.87 S)(1 - cos(42.75 D)) / M")
for D in (1, 2, 3, 4, 5, 6):
    for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8)):
        q = ok & (np.abs(pM - pS) == D) & ((M - S) / M > lo) & ((M - S) / M <= hi)
        if q.sum() < 300: continue
        sm = (s_all[ar, pS] - s_all[ar, pM])[q] / M[q]
        mm = (m_all[ar, pS] - m_all[ar, pM])[q] / M[q]
        vm = (v_all[ar, pS] - v_all[ar, pM])[q] / M[q]
        pred = (60 * (M - 0.87 * S) * (1 - np.cos(np.radians(42.75 * D))) / M)[q]
        print(f"  D={D} g in ({lo:.2f},{hi:.2f}]: s-margin {med(sm)}   m-part {med(mm)}   V-part {med(vm)}   V-pred {np.median(pred):6.1f}   P(s-margin>0) {np.mean(sm>0):.3f}  n={q.sum()}")

print("\nm-scores of non-max, non-runner-up inputs relative to the runner-up's m-score, in units of M, vs (x_p - S)/M  (age >= 2)")
rows_v, rows_d, rows_s = [], [], []
for p in range(0, 8):
    q = ok & (pM != p) & (pS != p)
    rows_v.append(((x[:, p] - S) / M)[q]); rows_d.append(((m_all[:, p] - m_all[ar, pS]) / M)[q]); rows_s.append(((s_all[:, p] - s_all[ar, pS]) / M)[q])
v = np.concatenate(rows_v); d = np.concatenate(rows_d); sd = np.concatenate(rows_s)
bins = [-9, -2, -1.5, -1, -0.7, -0.5, -0.35, -0.25, -0.15, -0.1, -0.05, -0.02, 0]
for lo, hi in zip(bins[:-1], bins[1:]):
    q = (v > lo) & (v <= hi)
    print(f"  v in ({lo:5.2f},{hi:5.2f}]: m-diff {med(d[q])}    s-diff {med(sd[q])}   n={q.sum()}")
