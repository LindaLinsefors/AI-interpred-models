"""E14. How the provisional scores l_t = U h_t change at events.  idx(p, t) = 9 - (t - p).

(N) new runner-up x_t in (S0, M): lead of the new input over the old runner-up vs (x_t - S0).
(R) new max at t: change of [score(old max) - score(old runner-up)] from t-1 to t ('deposit'),
    against the old gap M0 - S0 and M0, by age g of the old max and by |p_M0 - p_S0|; the latter also
    split by the age aS = t - p_S0 of the old runner-up at the record (aS = 1: it arrived just before the record).
Then persistence of score(old max) - score(old runner-up) k steps later, for the records with g >= 1.
The records that can be followed for k steps are a different set at every k, so that is not a decay curve.
"""
from common import *

x = sample(600_000, 1414)
H = run(x)
st = running_stats(x)
B = len(x)
ar = np.arange(B)
L = H @ U.T                                        # (B, T, 10) provisional logits at every step

def sc(t, p):                                      # provisional score of position p at time t
    return L[ar, t, 9 - (t - p)]

def lsq(y, cols):
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2

print("=== (N) new runner-up at t (S0 < x_t < M0, S0 > 0.2): score(t) - score(old runner-up) ~ k1 (x_t - S0) + k2 M")
for t in range(2, 10):
    M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
    s = (x[:, t] > S0) & (x[:, t] < M0) & (S0 > 0.2)
    pS0 = st["p2"][:, t - 1]
    y = (sc(t, np.full(B, t)) - sc(t, pS0))[s]
    b, r2 = lsq(y, [(x[:, t] - S0)[s], M0[s]])
    b1, r21 = lsq(y, [(x[:, t] - S0)[s]])
    print(f"  t={t}: k1 = {b[0]:6.1f}, k2 = {b[1]:5.2f}, R2 = {r2:.3f}   (slope-only fit: {b1[0]:6.1f}, R2 = {r21:.3f})   n = {s.sum()}")

print("\n=== (R) new max at t (M0 > 0.3, S0 > 0.1): deposit D = change of [score(old max) - score(old runner-up)] from t-1 to t")
print("    fit D ~ c1 (M0 - S0) + c2 M0 ; by age g of the old max at t-1")
M0a, S0a, Da, ga, sepa, da_, afa, pre, aSa = [], [], [], [], [], [], [], [], []
for t in range(2, 10):
    M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
    s = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1)
    pM0, pS0 = st["p1"][:, t - 1], st["p2"][:, t - 1]
    before = sc(t - 1, pM0) - sc(t - 1, pS0)
    after = sc(t, pM0) - sc(t, pS0)
    M0a.append(M0[s]); S0a.append(S0[s]); Da.append((after - before)[s]); ga.append((t - 1 - pM0)[s])
    sepa.append(np.abs(pM0 - pS0)[s]); da_.append((x[:, t] - M0)[s]); afa.append(after[s]); pre.append(before[s])
    aSa.append((t - pS0)[s])                           # age of the old runner-up at the record step
M0a, S0a, Da, ga, sepa, da_, afa, pre, aSa = map(np.concatenate, (M0a, S0a, Da, ga, sepa, da_, afa, pre, aSa))
for g in range(0, 8):
    s = ga == g
    if s.sum() < 300: continue
    b, r2 = lsq(Da[s], [(M0a - S0a)[s], M0a[s]])
    b3, r23 = lsq(afa[s], [(M0a - S0a)[s], M0a[s]])
    print(f"  g={g}: D ~ {b[0]:6.1f} (M0-S0) + {b[1]:6.2f} M0  (R2 {r2:.3f})   |  AFTER the record: score(old max)-score(old S) ~ "
          f"{b3[0]:6.1f} (M0-S0) + {b3[1]:6.2f} M0  (R2 {r23:.3f})   n={s.sum()}")
# split by aS = t - p_S0, the age of the old runner-up at the record: aS = 1 means it arrived on the step just before the record
for label, sub in (("all g >= 1", ga >= 1), ("g >= 1 and aS >= 2", (ga >= 1) & (aSa >= 2)), ("g >= 1 and aS = 1", (ga >= 1) & (aSa == 1))):
    print(f"    by separation |p_M0 - p_S0| ({label}): AFTER ~ c1 (M0 - S0) + c2 M0;  last column: c1 / (1 - cos(42.75 deg * sep))")
    for sp in range(1, 9):
        s = (sepa == sp) & sub
        if s.sum() < 300: continue
        b3, r23 = lsq(afa[s], [(M0a - S0a)[s], M0a[s]])
        print(f"      sep={sp}: {b3[0]:6.1f} (M0-S0) + {b3[1]:6.2f} M0   R2 {r23:.3f}   P(old max ahead of old S after record) = {(afa[s] > 0).mean():.3f}  n={s.sum():6d}"
              f"   c1/(1-cos) = {b3[0] / (1 - np.cos(np.radians(42.75 * sp))):6.1f}")
print("    dependence of AFTER on the new max's excess d = x_t - M0:  corr =", np.corrcoef(afa, da_)[0, 1])

print("\n=== persistence: for such records at t (g >= 1) with no later change of the top two, [score(old max) - score(old S)] k steps later ~ c1 (M0-S0) + c2 M0")
print("    (a different set of records at every k, so this is not a decay curve)")
# NB the set of records changes with k: a record followed for k steps must occur at t <= 9 - k, so larger k keeps earlier
# records with smaller separations, and c1 depends strongly on the separation. The sequence of c1 over k is not a decay curve.
for k in range(0, 6):
    ys, g1, g2 = [], [], []
    for t in range(2, 10 - k):
        M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
        pM0, pS0 = st["p1"][:, t - 1], st["p2"][:, t - 1]
        s = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1) & (t - 1 - pM0 >= 1)     # g >= 1, as in the separation tables above
        # top two unchanged between t and t+k
        s &= (st["p1"][:, t + k] == t) & (st["p2"][:, t + k] == st["p1"][:, t - 1])
        ys.append((sc(t + k, pM0) - sc(t + k, pS0))[s]); g1.append((M0 - S0)[s]); g2.append(M0[s])
    y, g1, g2 = map(np.concatenate, (ys, g1, g2))
    b, r2 = lsq(y, [g1, g2])
    print(f"  k={k}: {b[0]:6.1f} (M0-S0) + {b[1]:6.2f} M0   R2 {r2:.3f}   P(> 0) = {(y > 0).mean():.3f}   n={len(y)}")
