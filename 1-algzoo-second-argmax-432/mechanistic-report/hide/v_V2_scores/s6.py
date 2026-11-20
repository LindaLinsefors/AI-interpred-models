"""S6: new maximum at step t (x_t > M_{t-1}, M_{t-1} > 0.3, p1(t-1) < t-1), t = 3..9.
Compare s_t on inputs 0..t-2 (indices 9-t..7) with m_{t-1} on the same inputs (indices 10-t..8), and with
s_{t-1}: pooled R^2 with a common slope after removing each state's mean over those inputs.
P(old max is the largest entry of s_t among inputs 0..t-2).
No event (x_t < S_{t-1}, S_{t-1} > 0.1, all four comparators on at t): s_t vs shifted s_{t-1}.

Usage: python -B s6.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LEAD_R2 = [0.96, 0.77, 0.77, 0.88, 0.91, 0.89, 0.87]
LEAD_P = [0.996, 0.994, 0.991, 0.990, 0.988, 0.982, 0.964]
LEAD_NO = [0.67, 0.72, 0.80, 0.89, 0.93, 0.94, 0.96]
ts = range(3, 10)


class Acc:
    """Pooled common-slope fit y ~ b x (both already centred per state): accumulates sums."""
    def __init__(self):
        self.xx = self.xy = self.yy = 0.0; self.n = 0

    def add(self, X, Y):
        X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
        self.xx += (X * X).sum(); self.xy += (X * Y).sum(); self.yy += (Y * Y).sum(); self.n += len(X)

    @property
    def r2(self):
        return self.xy ** 2 / (self.xx * self.yy)

    @property
    def slope(self):
        return self.xy / self.xx


new_m = {t: Acc() for t in ts}; new_s = {t: Acc() for t in ts}
new_mt = {t: Acc() for t in ts}                      # m_t (instead of s_t) vs shifted m_{t-1} at a new max
no_s = {t: Acc() for t in ts}; no_s_all = {t: Acc() for t in ts}; no_m = {t: Acc() for t in ts}
no_s_noon = {t: Acc() for t in ts}                   # no event, without the "all four on" condition
hit = {t: 0 for t in ts}; hit_shift_m = {t: 0 for t in ts}; n_new = {t: 0 for t in ts}
allzero = {t: 0 for t in ts}; n_comp_on = {t: np.zeros(5) for t in ts}

for x, H, M, S, p1, p2 in batches(nb, seed0):
    for t in ts:
        s_t, V_t, m_t = scores(H[:, t]); s_p, V_p, m_p = scores(H[:, t - 1])
        new = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0.3) & (p1[:, t - 1] < t - 1)
        Y = s_t[new][:, 9 - t:8]; Xm = m_p[new][:, 10 - t:9]; Xs = s_p[new][:, 10 - t:9]
        new_m[t].add(Xm, Y); new_s[t].add(Xs, Y); new_mt[t].add(Xm, m_t[new][:, 9 - t:8])
        n_new[t] += new.sum()
        hit[t] += (Y.argmax(1) == p1[new, t - 1]).sum()
        hit_shift_m[t] += (Xm.argmax(1) == p1[new, t - 1]).sum()
        con = (H[new, t][:, COMP] > 0).sum(1)
        allzero[t] += (con == 0).sum(); n_comp_on[t] += np.bincount(con, minlength=5)
        on = (H[:, t][:, COMP] > 0).all(1)
        noev = (x[:, t] < S[:, t - 1]) & (S[:, t - 1] > 0.1)
        c = noev & on
        no_s[t].add(s_p[c][:, 10 - t:9], s_t[c][:, 9 - t:8])              # inputs 0..t-2
        no_s_all[t].add(s_p[c][:, 10 - t:], s_t[c][:, 9 - t:9])           # inputs 0..t-1 (all of s_{t-1}'s inputs)
        no_m[t].add(m_p[c][:, 10 - t:9], m_t[c][:, 9 - t:8])
        no_s_noon[t].add(s_p[noev][:, 10 - t:9], s_t[noev][:, 9 - t:8])

print(f"S6  seeds {seed0}..{seed0 + nb - 1}, {nb * B_DEFAULT} sequences")
print("NEW MAX at t:  n | R^2(s_t ~ shifted m_{t-1}) [lead] slope | R^2(s_t ~ shifted s_{t-1}) slope | P(old max = argmax s_t on inputs 0..t-2) [lead] se"
      " | P(old max = argmax of shifted m_{t-1}) | P(all 4 comparators zero at t) | R^2(m_t ~ shifted m_{t-1})")
for i, t in enumerate(ts):
    p = hit[t] / n_new[t]
    print(f"t={t}  {n_new[t]:7d} | {new_m[t].r2:.3f} [{LEAD_R2[i]:.2f}] b={new_m[t].slope:.3f} | {new_s[t].r2:.3f} b={new_s[t].slope:+.3f} | "
          f"{p:.4f} [{LEAD_P[i]:.3f}] {np.sqrt(p * (1 - p) / n_new[t]):.4f} | {hit_shift_m[t] / n_new[t]:.4f} | "
          f"{allzero[t] / n_new[t]:.4f} | {new_mt[t].r2:.3f}")
print("number of comparators on at the step of a new max (0..4), fractions:")
for t in ts:
    print(f"t={t}  ", " ".join(f"{v:.4f}" for v in n_comp_on[t] / n_comp_on[t].sum()))
print("\nNO EVENT at t (x_t < S_{t-1}, S_{t-1} > 0.1, all four on at t):  n | R^2(s_t ~ shifted s_{t-1}) on inputs 0..t-2 [lead] slope"
      " | same on inputs 0..t-1 | R^2(m_t ~ shifted m_{t-1}) | s_t ~ shifted s_{t-1} without the 'all four on' condition (n)")
for i, t in enumerate(ts):
    print(f"t={t}  {no_s[t].n:7d} | {no_s[t].r2:.3f} [{LEAD_NO[i]:.2f}] b={no_s[t].slope:.3f} | {no_s_all[t].r2:.3f} b={no_s_all[t].slope:.3f} | "
          f"{no_m[t].r2:.3f} b={no_m[t].slope:.3f} | {no_s_noon[t].r2:.3f} ({no_s_noon[t].n})")
