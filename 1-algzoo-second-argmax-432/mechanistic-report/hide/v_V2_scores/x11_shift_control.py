"""Off-by-one control for S6: is the best alignment between s_t (new max) and m_{t-1} a shift by exactly one index?
Y = s_t on indices 9-t..7 (inputs 0..t-2);  X_k = m_{t-1} on indices 9-t+k .. 7+k for k = 0, 1, 2
(k = 1 is the claimed 'same inputs' alignment). Same pooled common-slope R^2 after removing each state's mean.
Also the same for the no-event law (s_t vs s_{t-1}) and corr(A(d), B(d)) for S3.

Usage: python -B x11_shift_control.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 6100
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ts = range(3, 10)
acc = {(t, k, ev): np.zeros(3) for t in ts for k in (0, 1, 2) for ev in ("new", "quiet")}


def add(key, X, Y):
    X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
    acc[key] += [(X * X).sum(), (X * Y).sum(), (Y * Y).sum()]


for x, H, M, S, p1, p2 in batches(nb, seed0):
    for t in ts:
        s_t = scores(H[:, t])[0]; s_p, _, m_p = scores(H[:, t - 1])
        new = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0.3) & (p1[:, t - 1] < t - 1)
        quiet = (x[:, t] < S[:, t - 1]) & (S[:, t - 1] > 0.1) & (H[:, t][:, COMP] > 0).all(1)
        for k in (0, 1, 2):
            add((t, k, "new"), m_p[new][:, 9 - t + k:8 + k], s_t[new][:, 9 - t:8])
            add((t, k, "quiet"), s_p[quiet][:, 9 - t + k:8 + k], s_t[quiet][:, 9 - t:8])
print(f"seeds {seed0}..{seed0 + nb - 1}")
for ev, lab in (("new", "new max: s_t vs m_{t-1} shifted by k indices"), ("quiet", "no event: s_t vs s_{t-1} shifted by k indices")):
    print(f"\n{lab}: R^2 for k = 0, 1, 2 (k = 1 is the claim)")
    for t in ts:
        print(f"t={t}: " + "  ".join(f"k={k}: {acc[(t, k, ev)][1] ** 2 / (acc[(t, k, ev)][0] * acc[(t, k, ev)][2]):.3f}" for k in (0, 1, 2)))

A = np.array([-32.6, 13.7, 61.0, 81.9, 59.2, 7.1, -41.5, -62.3, -48.3, -5.8, 39.7, 59.7, 47.3, 13.8, -24.8])
B = np.array([3.9, 2.1, 6.4, 10.6, 10.8, 6.5, 0.7, -3.0, -2.4, 3.6, 10.4, 13.3, 11.0, 4.7, -2.9])
sl = np.polyfit(A, B, 1)
print(f"\nS3: correlation between my A(d) and B(d) over d = -7..7 (values from s2_s3_seed5000.txt): {np.corrcoef(A, B)[0, 1]:.3f};"
      f"  B ~ {sl[1]:.2f} + {sl[0]:.3f} A")
