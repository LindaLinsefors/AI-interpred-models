"""E15b. Refinement of E15 (i) and (iii): leave the newest input (index 9) out of the m-readout,
and at a new max compare only inputs that were at least one step old before the record."""
from common import *

x = sample(500_000, 1516)
H, Z = run(x, return_pre=True)
st = running_stats(x)
B = len(x)
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
S_all = H @ U.T
M_all = S_all - H @ Uc.T

print("(i) m_t = readout without comparators, restricted to inputs older than the current one (indices 9-t..8).")
print("    Conditioned on S_t > 0 and on the max of the prefix not being the current input:")
for t in range(2, 10):
    valid = (np.arange(10) >= 9 - t) & (np.arange(10) <= 8)
    ok = (st["M2"][:, t] > 0) & (st["p1"][:, t] < t)
    pm = np.where(valid, M_all[:, t], -np.inf).argmax(1) + t - 9
    ps = np.where(valid, S_all[:, t], -np.inf).argmax(1) + t - 9
    # runner-up among the older inputs only (prefix without the current input) for reference
    print(f"   t={t}: argmax m = max {np.mean(pm[ok] == st['p1'][ok, t]):.4f}   argmax m = runner-up {np.mean(pm[ok] == st['p2'][ok, t]):.4f}"
          f"   | argmax s (older inputs) = max {np.mean(ps[ok] == st['p1'][ok, t]):.4f}")

print("\n(iii) at a new max at step t: s_t on inputs 0..t-2 versus shifted m_(t-1) and shifted s_(t-1) (mean over those inputs removed)")
def r2(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    sl = (a * b).sum() / (b * b).sum()
    return 1 - ((a - sl * b) ** 2).sum() / (a ** 2).sum(), sl
for t in range(3, 10):
    s = (x[:, t] > st["M1"][:, t - 1]) & (st["M1"][:, t - 1] > 0.3) & (st["p1"][:, t - 1] < t - 1)
    new = S_all[s, t][:, 9 - t: 8]                 # inputs 0..t-2 at time t
    old_m = M_all[s, t - 1][:, 10 - t: 9]          # the same inputs at time t-1
    old_s = S_all[s, t - 1][:, 10 - t: 9]
    a, b = r2(new, old_m); c, d = r2(new, old_s)
    # does the old max top s_t among inputs 0..t-2 ?
    top = new.argmax(1) + 0
    pold = st["p1"][s, t - 1]
    print(f"   t={t}: vs shifted m: R2 = {a:.3f} (slope {b:.2f})   vs shifted s: R2 = {c:.3f} (slope {d:.2f})   "
          f"P(old max tops s_t among inputs 0..t-2) = {np.mean(top == pold):.3f}   n = {s.sum()}")

print("\n(iv) no event (x_t below S_(t-1), S_(t-1) > 0.1, all four comparators on at step t; step t-1 is not conditioned on): s_t on inputs 0..t-2 versus shifted s_(t-1)")
for t in range(3, 10):
    s = (x[:, t] < st["M2"][:, t - 1]) & (st["M2"][:, t - 1] > 0.1) & (Z[:, t][:, COMP] > 0).all(1)
    new = S_all[s, t][:, 9 - t: 8]
    old_s = S_all[s, t - 1][:, 10 - t: 9]
    c, d = r2(new, old_s)
    print(f"   t={t}: R2 = {c:.3f} (slope {d:.2f})   n = {s.sum()}")
