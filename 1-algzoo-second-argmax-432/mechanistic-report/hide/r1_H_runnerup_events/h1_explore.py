"""H1 exploration: the comparator thresholds at an N event, in units of the gap.

rho_j = (theta_j(t) - S_old) / (M_old - S_old)   so that comparator j is OFF iff r > rho_j,
r = (x_t - S_old)/(M_old - S_old).  Unit 4 is ON iff r > rho_4.
"""
from hcommon import *

x = sample(400_000, 1001)
e = Ev(x)
Mo, So, a1, a2, tt = e.Mo, e.So, e.a1, e.a2, e.tt
ok = (tt >= 2) & (So > 0) & (Mo > So)          # clean population: both old values positive
isN = ok & (x > So) & (x < Mo)
print("fraction of steps (t>=2) that are N events with S_old>0:", isN[:, 2:].mean(), " n =", isN.sum())
gap = np.where(ok, Mo - So, np.nan)
r = (x - So) / gap
rho = {j: (e.th[j] - So) / gap for j in COMP + [4]}

# thresholds are functions of h_{t-1} only, so describe them on ALL steps with ok (not only N events)
print("\n=== rho_j = (theta_j - S)/(M - S): median [25%, 75%] by age of the max a1 (all steps t>=2, S_old>0, gap/M>0.1) ===")
big = ok & (gap / Mo > 0.1)
for j in COMP + [4]:
    print(f" unit {j}")
    for a in range(1, 10):
        s = big & (a1 == a)
        if s.sum() < 500: continue
        v = rho[j][s]
        print(f"   a1={a}: {np.median(v):6.3f} [{np.percentile(v,25):6.3f}, {np.percentile(v,75):6.3f}]  n={s.sum()}")

print("\n=== same, split by whether the old runner-up is older (old max, a2>a1) or younger (a2<a1) than the max ===")
for j in COMP + [4]:
    print(f" unit {j}:  a1 | a2>a1 (S is a former max) | a2<a1 (S arrived after the max)")
    for a in range(1, 10):
        row = f"   a1={a}: "
        for cond in (a2 > a1, a2 < a1):
            s = big & (a1 == a) & cond
            row += (f"{np.median(rho[j][s]):6.3f} [{np.percentile(rho[j][s],25):6.3f}, {np.percentile(rho[j][s],75):6.3f}] n={s.sum():6d} | " if s.sum() > 300 else "      .                       | ")
        print(row)

print("\n=== regression theta_j(t) ~ bM M_old + bS S_old (no intercept), per a1; and with a2 split ===")
for j in COMP + [4]:
    print(f" unit {j}")
    for a in range(1, 10):
        s = ok & (a1 == a)
        if s.sum() < 500: continue
        b, R2 = lsq(e.th[j][s], [Mo[s], So[s]])
        print(f"   a1={a}: bM={b[0]:6.3f} bS={b[1]:6.3f} R2={R2:.4f}  (bM+bS={b.sum():.3f})")
