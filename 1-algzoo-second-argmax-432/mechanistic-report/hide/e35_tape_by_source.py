"""E35. The tape piece of the scores split exactly by source input.
The network is piecewise linear and bias-free, so h_9 = sum_p J_p x_p with J_p the exact trace of input p
(gates of this sequence held fixed). The tape piece is linear in h_9, hence
    s_tape[i] = sum_p kappa[i, p] x_p,     kappa[i, p] = P_tape[i] . J_p .
Question: which inputs make entry i, and with what kernel kappa?
Same selection as e33/e34 (final states, max and runner-up at least two steps old, M > 0.3, S > 0.1).
Run:  python -B hide/e35_tape_by_source.py > hide/outputs/e35_tape_by_source.txt"""
from common import *
from modes import RIGHT, LEFT

B = 150_000
x = sample(B, 3535)
H, J = traces(x)
Uo = U[:, ORD]
P = np.real(np.outer(Uo @ RIGHT["alt"], LEFT["alt"]))
for n in ("short", "h2", "h3"):
    P = P + 2 * np.real(np.outer(Uo @ RIGHT[n], LEFT[n]))            # (10, 13): s_tape = P h[o]
kap = np.einsum("io,bpo->bip", P, J[:, 9][:, :, ORD])                 # kappa[b, i, p]
C = kap * x[:, None, :]                                               # contribution of input p to entry i
G = H[:, 9][:, ORD] @ P.T
print("exactness: max |sum_p C - s_tape| =", float(np.abs(C.sum(2) - G).max()))
order = np.argsort(-x, 1)
pM, pS = order[:, 0], order[:, 1]
ar = np.arange(B)
M, S = x[ar, pM], x[ar, pS]
ok = (pM <= 7) & (pS <= 7) & (M > 0.3) & (S > 0.1)
print(ok.sum(), "states selected")
med = lambda v: float(np.median(v)) if len(v) else float("nan")

print("\n1. Entry i minus the runner-up's entry, D[i] = sum_p (C[i,p] - C[i_S,p]), split by which input p it comes from (units of M)")
Dp = (C - C[ar, pS][:, None, :]) / M[:, None, None]                   # (B, i, p)
own = Dp[:, np.arange(10), np.arange(10)]                             # p = i
fromM, fromS = Dp[ar, :, pM], Dp[ar, :, pS]
tot = Dp.sum(2)
other = tot - own - fromM - fromS
idx = np.arange(10)[None, :]
u = (S[:, None] - x) / M[:, None]
base = ok[:, None] & (idx <= 7) & (idx != pM[:, None]) & (idx != pS[:, None])
print("   group of entries              n      total |  own input   the maximum   the runner-up   all other inputs   (median; std in brackets)")
for name, q in (("below, u < 0.05", base & (u > 0) & (u < 0.05)), ("below, 0.05 < u < 0.2", base & (u > 0.05) & (u < 0.2)),
                ("below, 0.2 < u < 0.5", base & (u > 0.2) & (u < 0.5)), ("below, u > 0.5", base & (u > 0.5))):
    print(f"   {name:24s} {q.sum():8d}  {med(tot[q]):6.1f} | " + "   ".join(f"{med(v[q]):6.1f} ({v[q].std():5.1f})" for v in (own, fromM, fromS, other)))
qm = ok
tm, om, sm_ = tot[ar, pM][qm], fromM[ar, pM][qm], fromS[ar, pM][qm]
print(f"   the maximum's own entry  {qm.sum():8d}  {med(tm):6.1f} |  from the maximum {med(om):6.1f} ({om.std():5.1f})   from the runner-up {med(sm_):6.1f} ({sm_.std():5.1f})   "
      f"others {med(tm - om - sm_):6.1f} ({(tm - om - sm_).std():5.1f})")

def kernel(name, pin, sel, ages):
    print(f"\n{name}: median kappa[i, p] (logits per unit of that input), rows = age of the input, columns = i - p (entries of age >= 2 only)")
    print("   age      n   " + " ".join(f"{d:+6d}" for d in range(-7, 8)))
    for a in ages:
        q = sel & (pin == 9 - a)
        if q.sum() < 300:
            continue
        row = []
        for dlt in range(-7, 8):
            i = 9 - a + dlt
            row.append(f"{med(kap[ar[q], i, pin[q]]):6.1f}" if 0 <= i <= 7 else "     .")
        print(f"   {a}  {q.sum():6d}   " + " ".join(row))

kernel("2. Kernel of the maximum", pM, ok, range(2, 10))
kernel("3a. Kernel of the runner-up, when it arrived after the maximum", pS, ok & (pS > pM), range(2, 10))
kernel("3b. Kernel of the runner-up, when it arrived before the maximum", pS, ok & (pS < pM), range(2, 10))
pT = order[:, 2]
T3 = x[ar, pT]
kernel("4a. Kernel of the third largest, within 0.1 M of the runner-up", pT, ok & (pT <= 7) & (S - T3 < 0.1 * M) & (T3 > 0), range(2, 10))
kernel("4b. Kernel of the third largest, more than 0.4 M below the runner-up", pT, ok & (pT <= 7) & (S - T3 > 0.4 * M), range(2, 10))
