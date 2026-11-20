"""Follow-ups to c03 / c04, and the same 'pooled over order' question for e16 (claim S5).

 (1) e14 / D2: fixed cohorts, per-sequence ratios y(k)/y(0), and c1 within one separation.
 (2) e18 / I2: P(answer = p0 - k) among sequences in which that position exists and is not the old
     runner-up q0.
 (3) e16 / S5: V-part, m-part and s-margin split by order (runner-up earlier / later than the max).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import LEFT, RIGHT, LAM

# ---------------------------------------------------------------- (1)
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
    return b, 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print("(1) D2 with fixed cohorts (records with M0 > 0.3, S0 > 0.1, g >= 1, top two unchanged through t + K)")
for K, tmax in ((5, 4), (3, 6), (2, 7)):
    Y = {k: [] for k in range(K + 1)}; G1, G2, SEP = [], [], []
    for t in range(2, tmax + 1):
        M0, S0 = st["M1"][:, t - 1], st["M2"][:, t - 1]
        pM0, pS0 = st["p1"][:, t - 1], st["p2"][:, t - 1]
        s = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1) & (t - 1 - pM0 >= 1)
        s &= (st["p1"][:, t + K] == t) & (st["p2"][:, t + K] == pM0)
        for k in range(K + 1):
            Y[k].append((sc(t + k, pM0) - sc(t + k, pS0))[s])
        G1.append((M0 - S0)[s]); G2.append(M0[s]); SEP.append(np.abs(pM0 - pS0)[s])
    G1, G2, SEP = map(np.concatenate, (G1, G2, SEP))
    Y = {k: np.concatenate(v) for k, v in Y.items()}
    print(f"   cohort K = {K} (records at t = 2..{tmax}), n = {len(G1)}; separations present: " +
          ", ".join(f"{sp}: {np.mean(SEP == sp):.2f}" for sp in range(1, 8) if (SEP == sp).any()))
    print("      k | c1, c2 (two regressors) | slope of y on (M0 - S0) alone | median y / M0 | median of y(k) / y(0) | P(y > 0)")
    for k in range(K + 1):
        b, r2 = lsq(Y[k], [G1, G2]); b1, _ = lsq(Y[k], [G1])
        pos = Y[0] > 1e-6
        print(f"      {k} | {b[0]:5.1f} {b[1]:5.2f} | {b1[0]:5.1f} | {np.median(Y[k] / G2):5.2f} | {np.median(Y[k][pos] / Y[0][pos]):.3f} | {np.mean(Y[k] > 0):.3f}")
    for sp in (1, 2, 3, 4):
        q = SEP == sp
        if q.sum() < 2000:
            continue
        print(f"      separation {sp} only (n = {q.sum()}): c1 by k: " + "  ".join(f"{lsq(Y[k][q], [G1[q], G2[q]])[0][0]:5.1f}" for k in range(K + 1))
              + "   | median y/M0 by k: " + "  ".join(f"{np.median(Y[k][q] / G2[q]):5.2f}" for k in range(K + 1)))
del L

# ---------------------------------------------------------------- (2)
print("\n(2) e18 rotations, conditional on the target position p0 - k existing (0 <= p0 - k <= tr - 1) and not being the old runner-up q0")
x = sample(1_500_000, 1818)
st = running_stats(x)
truth = second_argmax(x)
B = len(x); ar = np.arange(B)
om = np.angle(LAM["clock"])
tr = x.argmax(1)
ok = tr >= 3
p0 = np.where(ok, st["p1"][ar, np.maximum(tr - 1, 0)], -1)
q0 = np.where(ok, st["p2"][ar, np.maximum(tr - 1, 0)], -1)
M0 = st["M1"][ar, np.maximum(tr - 1, 0)]; S0 = st["M2"][ar, np.maximum(tr - 1, 0)]
g = tr - 1 - p0
ok &= (g >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0) & (tr <= 8)
xs, trs, p0s, q0s = x[ok], tr[ok], p0[ok], q0[ok]
del st


def run_mod(x, tr, fn, clip=False):
    h = np.zeros((len(x), N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        sel = np.where(tr - 1 == t)[0]
        if len(sel) and fn is not None:
            z = h[sel][:, ORD] @ LEFT["clock"]
            hn = h[sel][:, ORD] + 2 * np.real(np.multiply.outer(fn(z) - z, RIGHT["clock"]))
            h[np.ix_(sel, ORD)] = np.maximum(hn, 0) if clip else hn
    return (h @ U.T).argmax(1)


print("     k | e18's P(answer = p0 - k) | share with target existing | share with target = q0 | P(answer = p0 - k | exists, != q0) | same, clipped edit | P(answer = p0 | exists, != q0) | P(answer = q0 | exists, != q0)")
for k in (-2, -1, 1, 2):
    a = run_mod(xs, trs, lambda z, k=k: z * np.exp(1j * om * k))
    c = run_mod(xs, trs, lambda z, k=k: z * np.exp(1j * om * k), clip=True)
    tgt = p0s - k
    ex = (tgt >= 0) & (tgt <= trs - 1)
    q = ex & (q0s != tgt)
    print(f"    {k:+d} | {np.mean(a == tgt):.3f} | {ex.mean():.3f} | {np.mean(q0s == tgt):.3f} | {np.mean(a[q] == tgt[q]):.3f} | {np.mean(c[q] == tgt[q]):.3f} | {np.mean(a[q] == p0s[q]):.3f} | {np.mean(a[q] == q0s[q]):.3f}   n = {q.sum()}")

# ---------------------------------------------------------------- (3)
print("\n(3) e16 / S5 split by order.  medians in units of M of (V-part, m-part, s-margin); 'S earlier' = runner-up before the max (an overtaken max), 'S later' = runner-up after the max")
x = sample(800_000, 1616)
H, Z = run(x, return_pre=True)
st = running_stats(x)
B = len(x); ar = np.arange(B); t = 9
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
s_all = H[:, t] @ U.T; v_all = H[:, t] @ Uc.T; m_all = s_all - v_all
M, S = st["M1"][:, t], st["M2"][:, t]
pM, pS = st["p1"][:, t], st["p2"][:, t]
ok = (Z[:, t][:, COMP] > 0).all(1) & (S > 0.2) & (pM < 9) & (pS < 9)
gap = (M - S) / M
for D, (lo, hi) in ((1, (0.3, 0.5)), (3, (0.05, 0.15)), (3, (0.15, 0.3)), (3, (0.3, 0.5)), (3, (0.5, 0.8)), (4, (0.3, 0.5)), (5, (0.3, 0.5)), (6, (0.3, 0.5))):
    row = []
    for lab, cond in (("pooled", np.ones(B, bool)), ("S earlier", pS < pM), ("S later", pS > pM)):
        q = ok & (np.abs(pM - pS) == D) & (gap > lo) & (gap <= hi) & cond
        vm = np.median((v_all[ar, pS] - v_all[ar, pM])[q] / M[q]); mm = np.median((m_all[ar, pS] - m_all[ar, pM])[q] / M[q])
        sm = (s_all[ar, pS] - s_all[ar, pM])[q] / M[q]
        row.append(f"{lab}: ({vm:5.1f}, {mm:6.1f}, {np.median(sm):5.1f}; P>0 {np.mean(sm > 0):.3f}; n={q.sum():5d})")
    pred = np.median((60 * (M - 0.87 * S) * (1 - np.cos(np.radians(42.75 * D))) / M)[ok & (np.abs(pM - pS) == D) & (gap > lo) & (gap <= hi)])
    print(f"   D={D} gap ({lo},{hi}]: " + "  |  ".join(row) + f"  | predicted V-part {pred:5.1f}")
