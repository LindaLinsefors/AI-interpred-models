"""E15. Two score vectors (after Zihao Chen's 2024 note): s_t = U h_t (all units) and
m_t = U h_t with the comparator columns 1,6,7,8 removed.  V_t = s_t - m_t is the comparators' vote.

(ii) the result of this script: the comparators' vote V as a function of position relative to the max:
     V[idx] ~ A(a1, idx) * (M - S) + B(a1, idx) * S + C(a1, idx) * x_t  on steps where all four comparators are on;
     is A a negative cosine bump centred on the max's index 9 - a1 ?  The same regression is run for m and for
     s = m + V, to see what is left of the bump in the sum.
(i) and (iii) are SUPERSEDED by e15b and kept only for the record; do not quote them:
(i)  accuracy of argmax m_t for the MAX of the prefix and of argmax s_t for the runner-up, per t.  Here the
     current input (index 9) is allowed in the argmax; e15b (i) leaves it out.
(iii) at a new max:  s_t (positions older than t)  vs  shifted m_{t-1}.  Here the input of step t-1 is included;
     e15b (iii) compares only inputs that were at least one step old before the record.
"""
from common import *

x = sample(500_000, 1515)
H, Z = run(x, return_pre=True)
st = running_stats(x)
B = len(x)
ar = np.arange(B)
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
S_all = H @ U.T
V_all = H @ Uc.T
M_all = S_all - V_all

print("(i) [SUPERSEDED by e15b (i): index 9, the current input, is allowed in the argmax here]")
print("    per step t: P(argmax m_t = max of prefix), P(argmax s_t = runner-up of prefix), on states with S_t > 0")
for t in range(1, 10):
    valid = np.arange(10) >= 9 - t
    sp = st["M2"][:, t] > 0
    pm = np.where(valid, M_all[:, t], -np.inf).argmax(1) + t - 9
    ps = np.where(valid, S_all[:, t], -np.inf).argmax(1) + t - 9
    print(f"   t={t}: m -> max {np.mean(pm[sp] == st['p1'][sp, t]):.4f}   m -> runner-up {np.mean(pm[sp] == st['p2'][sp, t]):.4f}"
          f"   s -> runner-up {np.mean(ps[sp] == st['p2'][sp, t]):.4f}   s -> max {np.mean(ps[sp] == st['p1'][sp, t]):.4f}")

print("\n(ii) comparators' vote at t = 9, steps with all four comparators on and S > 0.1:  V[idx] ~ A (M-S) + B S + C x_9")
t = 9
allon = (Z[:, t][:, COMP] > 0).all(1) & (st["M2"][:, t] > 0.1)
Mx, Sx, a1 = st["M1"][:, t], st["M2"][:, t], st["a1"][:, t]
print("  rows: age of max a1 (its index is 9 - a1);  columns: logit index 0..9;  entries: A(a1, idx)")
print("  R2 is centred per index and pooled over indices: over all ten, and over 0..7 only (index 9, V[9] ~ 58 x_9, fits almost exactly; its share of the pooled variance is the last number)")
Atab = {name: np.full((10, 10), np.nan) for name in "Vms"}      # coefficient of (M - S) in V, in m and in s = m + V
Btab = np.full((10, 10), np.nan); Ctab = np.full((10, 10), np.nan); R2 = np.full(10, np.nan)
for a in range(1, 10):
    s = allon & (a1 == a)
    X = np.column_stack([(Mx - Sx)[s], Sx[s], x[s, t]])
    Y = V_all[s, t]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Atab["V"][a], Btab[a], Ctab[a] = coef
    for name, Yo in (("m", M_all[s, t]), ("s", S_all[s, t])):   # the same regression for m and for the sum
        Atab[name][a] = np.linalg.lstsq(X, Yo, rcond=None)[0][0]
    res, tot = ((Y - X @ coef) ** 2).sum(0), ((Y - Y.mean(0)) ** 2).sum(0)
    R2[a] = 1 - res.sum() / tot.sum()
    print(f"   a1={a} (max at idx {9-a}): " + " ".join(f"{v:7.1f}" for v in coef[0])
          + f"   R2 = {R2[a]:.3f} (idx 0..9), {1 - res[:8].sum() / tot[:8].sum():.3f} (idx 0..7)  n={s.sum()}  idx 9: {tot[9] / tot.sum():.2f}")
print("  B(a1, idx) (coefficient of S):")
for a in range(1, 10):
    print(f"   a1={a}: " + " ".join(f"{v:7.1f}" for v in Btab[a]))
print("  fitted C(a1, idx = 9) (coefficient of x_9 at the newest index), a1 = 1..9: " + " ".join(f"{Ctab[a, 9]:.1f}" for a in range(1, 10)))
print("  fitted C(idx), mean over a1 = 2..9:", np.round(Ctab[2:].mean(0), 1), "  vs the weight product U[:, COMP] @ w[COMP] =", np.round(U[:, COMP] @ w[COMP], 1))

print("\n  A(a1, idx) as a function of d = idx - (9 - a1) (distance from the max's index), averaged over a1 = 2..9 with idx <= 7;")
print("  last two columns: the same coefficient of (M - S) in m = s - V (readout through the non-comparator units) and in the sum s = m + V")
byd = {name: {} for name in "Vms"}
for a in range(2, 10):
    for idx in range(0, 8):
        for name in "Vms":
            byd[name].setdefault(idx - (9 - a), []).append(Atab[name][a, idx])
for d in sorted(byd["V"]):
    v = np.array(byd["V"][d])
    print(f"    d = {d:+d}: mean A = {v.mean():7.1f}   (min {v.min():7.1f}, max {v.max():7.1f}, n = {len(v)})   -cos(42.75 d) * 60 = {-60*np.cos(np.radians(42.75*d)):7.1f}"
          f"   |   in m: {np.mean(byd['m'][d]):7.1f}   in s: {np.mean(byd['s'][d]):7.1f}")

print("\n(iii) [SUPERSEDED by e15b (iii): the input of step t-1 is included in the comparison here]")
print("    at a new max at step t (t >= 3, old max > 0.3): compare s_t at the indices of older inputs with the shifted m_{t-1} and shifted s_{t-1}")
for t in (3, 5, 7, 9):
    s = (x[:, t] > st["M1"][:, t - 1]) & (st["M1"][:, t - 1] > 0.3)
    new = S_all[s, t][:, 9 - t: 9]                         # indices 9-t .. 8  (inputs 0..t-1)
    old_m = M_all[s, t - 1][:, 10 - t: 10]                 # same inputs, one step earlier
    old_s = S_all[s, t - 1][:, 10 - t: 10]
    def r2(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        sl = (a * b).sum() / (b * b).sum()
        return 1 - ((a - sl * b) ** 2).sum() / (a ** 2).sum(), sl
    print(f"   t={t}: s_t vs shifted m_(t-1): R2 = {r2(new, old_m)[0]:.3f} (slope {r2(new, old_m)[1]:.2f});   "
          f"s_t vs shifted s_(t-1): R2 = {r2(new, old_s)[0]:.3f}   n = {s.sum()}")
