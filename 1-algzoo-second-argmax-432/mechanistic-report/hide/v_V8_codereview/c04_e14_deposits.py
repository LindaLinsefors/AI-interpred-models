"""Code review of e14_score_updates.py (claims D1, D2, D3).

 (a) D1: exact c1 / (1 - cos(42.75 deg * sep)); split by the order of the two positions (old max before /
     after the old runner-up) and by whether the record is at the last step; uncentred vs centred R2.
 (b) D2 ('persistence'): the claim says 'for such records' (D1 has g >= 1) but e14's loop has no g
     condition; and the population changes with k (t <= 9 - k, top two unchanged for k steps).
     Redo with g >= 1, and with ONE fixed cohort followed over k.
 (c) D3: nothing suspicious in the code; reproduce and add the uncentred R2 for reference.
Same sample as e14 (seed 1414).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

x = sample(600_000, 1414)
H = run(x)
st = running_stats(x)
B = len(x); ar = np.arange(B)
L = H @ U.T
del H


def sc(t, p):
    return L[ar, t, 9 - (t - p)]


def lsq(y, cols):
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = ((y - X @ b) ** 2).sum()
    return b, 1 - res / ((y - y.mean()) ** 2).sum(), 1 - res / (y ** 2).sum()


# ---------------------------------------------------------------- (a) D1
rec = []
for t in range(2, 10):
    M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
    s = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1)
    pM0, pS0 = st["p1"][:, t - 1], st["p2"][:, t - 1]
    after = sc(t, pM0) - sc(t, pS0)
    rec.append(np.column_stack([np.full(s.sum(), t), M0[s], S0[s], (t - 1 - pM0)[s], (pM0 - pS0)[s], after[s]]))
R = np.concatenate(rec)
tt, M0a, S0a, ga, dsg, afa = R.T
print("(a) D1 by separation, g >= 1 (as e14):  c1, c2, centred R2, uncentred R2, P(y > 0), c1 / (1 - cos(42.75 sep)) computed without rounding")
for sp in range(1, 8):
    s = (np.abs(dsg) == sp) & (ga >= 1)
    b, r2, r2u = lsq(afa[s], [(M0a - S0a)[s], M0a[s]])
    print(f"   sep={sp}: c1 = {b[0]:6.2f}  c2 = {b[1]:5.2f}  R2 = {r2:.3f}  (uncentred {r2u:.3f})  P(y>0) = {np.mean(afa[s] > 0):.3f}   "
          f"c1/(1-cos) = {b[0] / (1 - np.cos(np.radians(42.75 * sp))):6.1f}   n = {s.sum()}")
print("\n    split by order: old max EARLIER than old runner-up (pM0 < pS0) | old max LATER (pM0 > pS0);  c1, P(y>0), n")
for sp in range(1, 8):
    row = []
    for sign in (-1, 1):
        s = (dsg == sign * sp) & (ga >= 1)
        if s.sum() < 300:
            row.append("      .                      "); continue
        b, r2, _ = lsq(afa[s], [(M0a - S0a)[s], M0a[s]])
        row.append(f"c1 = {b[0]:6.1f}  c2 = {b[1]:5.2f}  R2 = {r2:.3f}  P = {np.mean(afa[s] > 0):.3f}  n = {s.sum():6d}")
    print(f"   sep={sp}: " + "  |  ".join(row))
print("\n    split by record step: t <= 7 | t = 8 | t = 9;  c1 (P(y>0))")
for sp in range(1, 8):
    row = []
    for lab, cond in (("t<=7", tt <= 7), ("t=8", tt == 8), ("t=9", tt == 9)):
        s = (np.abs(dsg) == sp) & (ga >= 1) & cond
        if s.sum() < 300:
            row.append("      .         "); continue
        b, r2, _ = lsq(afa[s], [(M0a - S0a)[s], M0a[s]])
        row.append(f"{b[0]:6.1f} ({np.mean(afa[s] > 0):.3f}, n={s.sum():6d})")
    print(f"   sep={sp}: " + "  |  ".join(row))

# ---------------------------------------------------------------- (b) D2
print("\n(b) D2, persistence of y = s[old max] - s[old runner-up] k steps after the record; c1 of y ~ c1 (M0 - S0) + c2 M0")
print("    k | e14 (all g >= 0, cohort changes with k) | g >= 1 (the population of D1) | fixed cohort: top two unchanged through t + 5, g >= 1, t <= 4")
fixed = {}
for k in range(0, 6):
    out = []
    for gmin in (0, 1):
        ys, g1, g2 = [], [], []
        for t in range(2, 10 - k):
            M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
            pM0, pS0 = st["p1"][:, t - 1], st["p2"][:, t - 1]
            s = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1) & (t - 1 - pM0 >= gmin)
            s &= (st["p1"][:, t + k] == t) & (st["p2"][:, t + k] == pM0)
            ys.append((sc(t + k, pM0) - sc(t + k, pS0))[s]); g1.append((M0 - S0)[s]); g2.append(M0[s])
        y, g1, g2 = map(np.concatenate, (ys, g1, g2))
        b, r2, _ = lsq(y, [g1, g2])
        out.append(f"c1 = {b[0]:5.1f}  c2 = {b[1]:5.2f}  R2 = {r2:.3f}  P(>0) = {np.mean(y > 0):.3f}  n = {len(y):6d}")
    ys, g1, g2 = [], [], []
    for t in range(2, 5):
        M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
        pM0, pS0 = st["p1"][:, t - 1], st["p2"][:, t - 1]
        s = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1) & (t - 1 - pM0 >= 1)
        s &= (st["p1"][:, t + 5] == t) & (st["p2"][:, t + 5] == pM0)
        ys.append((sc(t + k, pM0) - sc(t + k, pS0))[s]); g1.append((M0 - S0)[s]); g2.append(M0[s])
    y, g1, g2 = map(np.concatenate, (ys, g1, g2))
    b, r2, _ = lsq(y, [g1, g2])
    out.append(f"c1 = {b[0]:5.1f}  c2 = {b[1]:5.2f}  R2 = {r2:.3f}  P(>0) = {np.mean(y > 0):.3f}  n = {len(y):6d}")
    print(f"    {k} | " + " | ".join(out))

# ---------------------------------------------------------------- (c) D3
print("\n(c) D3, new runner-up at t: y = s_t[t] - s_t[old runner-up] ~ k1 (x_t - S0) + k2 M0;  k1, k2, centred R2, uncentred R2")
for t in range(2, 10):
    M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
    s = (x[:, t] > S0) & (x[:, t] < M0) & (S0 > 0.2)
    pS0 = st["p2"][:, t - 1]
    y = (sc(t, np.full(B, t)) - sc(t, pS0))[s]
    b, r2, r2u = lsq(y, [(x[:, t] - S0)[s], M0[s]])
    print(f"   t={t}: k1 = {b[0]:5.1f}  k2 = {b[1]:5.2f}  R2 = {r2:.3f}  (uncentred {r2u:.3f})  P(y > 0) = {np.mean(y > 0):.3f}  n = {s.sum()}")
