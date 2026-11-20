"""S6 follow-up: at a new max, is s_t (old inputs) really no better than the shifted m_{t-1}?
Split by the gap between the OLD max and the OLD runner-up, g = (M_{t-1} - S_{t-1}) / M_{t-1}
(events with S_{t-1} > 0 and the old runner-up among inputs 0..t-2).

Per gap bin: P(old max = argmax) for  (a) s_t,  (b) shifted m_{t-1},  (c) s_t after wiping the comparators at t-1.

Usage: python -B x9_new_max_by_gap.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5900
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
GAPS = [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]
ts = range(3, 10)
n = np.zeros((10, 6)); hs = np.zeros((10, 6)); hm = np.zeros((10, 6)); hi = np.zeros((10, 6))
ng = np.zeros((10, 6)); hsg = np.zeros((10, 6)); hmg = np.zeros((10, 6))          # by age g of the old max, gap <= 0.15
for x, H, M, S, p1, p2 in batches(nb, seed0):
    for t in ts:
        new = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0.3) & (p1[:, t - 1] < t - 1) & (S[:, t - 1] > 0) & (p2[:, t - 1] < t - 1)
        s_t = scores(H[new, t])[0][:, 9 - t:8]; m_p = scores(H[new, t - 1])[2][:, 10 - t:9]
        hp = H[new, t - 1].copy(); hp[:, COMP] = 0.0
        s_i = (np.maximum(x[new, t, None] * w + hp @ W.T, 0.0) @ U.T)[:, 9 - t:8]
        old = p1[new, t - 1]
        gb = np.digitize(((M[:, t - 1] - S[:, t - 1]) / M[:, t - 1])[new], GAPS, right=True) - 1
        np.add.at(n[t], gb, 1); np.add.at(hs[t], gb, s_t.argmax(1) == old); np.add.at(hm[t], gb, m_p.argmax(1) == old)
        np.add.at(hi[t], gb, s_i.argmax(1) == old)

print(f"seeds {seed0}..{seed0 + nb - 1}; new max at t, old max > 0.3 and at least 1 step old, old runner-up > 0 and among inputs 0..t-2")
print("gap bins of the OLD top two:", GAPS)
for nm, h in (("s_t (real)", hs), ("shifted m_{t-1}", hm), ("s_t, comparators wiped at t-1", hi)):
    print(f"\nP(old max is the largest entry among inputs 0..t-2) for {nm}:  rows t = 3..9, columns gap bins")
    for t in ts:
        print(t, " ".join(f"{h[t, k] / n[t, k]:.3f}" for k in range(6)), "   n:", " ".join(f"{int(n[t, k]):6d}" for k in range(6)))
    print("pooled", " ".join(f"{h[:, k].sum() / n[:, k].sum():.3f}" for k in range(6)), f"  all gaps: {h.sum() / n.sum():.4f}")
