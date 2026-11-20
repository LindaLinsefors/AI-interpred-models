"""E13. What are the provisional scores?  l_t = U h_t, index 9 - a = input of age a.
For every step t and every earlier-or-current position p: score(p) - score(runner-up), in units of M
(the net is homogeneous), against the input's value relative to the runner-up, (x_p - S)/M, by role
(max / third / other) and by age."""
from common import *

x = sample(400_000, 1313)
H = run(x)
st = running_stats(x)
B = len(x)
rows = []
for t in range(2, 10):
    l = H[:, t] @ U.T
    M, S = st["M1"][:, t], st["M2"][:, t]
    ok = (S > 0.2)                                  # runner-up clearly positive
    pS, pM, pT = st["p2"][:, t], st["p1"][:, t], st["p3"][:, t]
    lS = l[np.arange(B), 9 - (t - pS)]
    for p in range(t + 1):
        a = t - p
        role = np.where(p == pM, 0, np.where(p == pS, 1, np.where(p == pT, 2, 3)))
        d = (l[:, 9 - a] - lS) / M
        v = (x[:, p] - S) / M
        rows.append(np.column_stack([np.full(B, t), np.full(B, a), role, d, v, ok, M, S]))
R = np.concatenate(rows)
R = R[R[:, 5] == 1]
t_, a_, role, d, v = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]

print("score(p) - score(runner-up), in units of M; medians by relative value v = (x_p - S)/M")
bins = [-9, -2, -1.5, -1, -0.7, -0.5, -0.35, -0.25, -0.15, -0.1, -0.05, -0.02, 0]
print("\nNON-MAX inputs (third and others), all ages >= 2 (older than the previous input):")
print("   v bin            median   q25     q75      n      median/v")
s = (role >= 2) & (a_ >= 2)
for lo, hi in zip(bins[:-1], bins[1:]):
    q = s & (v > lo) & (v <= hi)
    if q.sum() < 200: continue
    med = np.median(d[q]); mv = np.median(v[q])
    print(f"  ({lo:5.2f},{hi:5.2f}]   {med:7.2f} {np.percentile(d[q],25):7.2f} {np.percentile(d[q],75):7.2f}  {q.sum():8d}   {med/mv:7.1f}")
for A in (0, 1):
    print(f"\nNON-MAX inputs at age {A}:")
    s = (role >= 2) & (a_ == A)
    for lo, hi in zip(bins[:-1], bins[1:]):
        q = s & (v > lo) & (v <= hi)
        if q.sum() < 200: continue
        med = np.median(d[q]); mv = np.median(v[q])
        print(f"  ({lo:5.2f},{hi:5.2f}]   {med:7.2f} {np.percentile(d[q],25):7.2f} {np.percentile(d[q],75):7.2f}  {q.sum():8d}   {med/mv:7.1f}")

print("\nTHE MAX: score(max) - score(runner-up) in units of M, by relative gap v = (M - S)/M and age of the max")
gb = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
print("  age | " + " ".join(f"({lo:.2f},{hi:.2f}]" for lo, hi in zip(gb[:-1], gb[1:])))
for A in range(10):
    row = []
    for lo, hi in zip(gb[:-1], gb[1:]):
        q = (role == 0) & (a_ == A) & (v > lo) & (v <= hi)
        row.append(f"{np.median(d[q]):11.1f}" if q.sum() > 200 else "          .")
    print(f"   {A}  | " + " ".join(row))
