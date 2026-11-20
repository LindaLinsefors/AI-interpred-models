"""Reviewer N1, extra checks.  Run: python -B n1_extra.py
(1) section 3.1: is the running answer 'the network's accuracy on zero-prefixed inputs'?  Compare the argmax
    restricted to existing positions with the unrestricted argmax over all ten indices.
(2) section 2: are all-off states ever anything but a new maximum?
(3) summary / panel (a): median phase of the clock coordinate z by age of the max and relative gap.
(4) section 3.2: the max's own entry over its first two steps, followed on the same sequences.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import coords

x = sample(400_000, 8201)
H, Z = run(x, return_pre=True)
st = running_stats(x)
M, S, p1, p2, a1, a2 = st["M1"], st["M2"], st["p1"], st["p2"], st["a1"], st["a2"]
B = len(x); ar = np.arange(B)
L = H @ U.T

print("(1) t: restricted acc | unrestricted acc | P(unrestricted top on a zero position)   [states with S_t > 0]")
for t in range(1, 10):
    sp = S[:, t] > 0
    valid = np.arange(10) >= 9 - t
    top_r = np.where(valid, L[:, t], -np.inf).argmax(1)
    top_u = L[:, t].argmax(1)
    tgt = 9 - a2[:, t]
    print(f"   {t}:  {np.mean(top_r[sp] == tgt[sp]):.4f} | {np.mean(top_u[sp] == tgt[sp]):.4f} | {np.mean(top_u[sp] < 9 - t):.4f}")

non = (Z[:, :, COMP] > 0).sum(-1)
newmax = p1 == np.arange(10)[None, :]
off = non == 0
print("\n(2) all-off states:", off.sum(), " of which not a new maximum:", (off & ~newmax).sum(),
      "  new maxima with x_t <= 0 among all-off:", (off & newmax & (x <= 0)).sum())

z = coords(H)["clock"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
g = (M - Sp) / np.where(M > 0, M, np.nan)
print("\n(3) median phase of z (deg) by age of the max 1..9, for bins of the relative gap (M - S+)/M, states with M > 0.3")
for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
    row = []
    for a in range(1, 10):
        s_ = (M > 0.3) & (a1 == a) & (g > lo) & (g <= hi)
        zz = z[s_] / M[s_]
        row.append(np.degrees(np.angle(np.median(zz.real) + 1j * np.median(zz.imag))) % 360)
    print(f"   gap ({lo}, {hi}]:", np.round(row, 1))

print("\n(4) the max's own entry relative to the runner-up's, in units of M, at age 0, 1, 2 on the same sequences")
print("    (max arrives at step p <= 7 with S_p > 0.1, next two inputs below the runner-up of the time)")
d0, d1, d2 = [], [], []
for p in range(1, 8):
    s_ = newmax[:, p] & (S[:, p] > 0.1) & (x[:, p + 1] < S[:, p]) & (x[:, p + 2] < S[:, p + 1])
    for k, dd in enumerate((d0, d1, d2)):
        t = p + k
        lS = L[ar, t, 9 - (t - p2[:, t])]
        dd.append(((L[:, t, 9 - k] - lS) / M[:, t])[s_])
print("    medians:", [round(float(np.median(np.concatenate(d))), 2) for d in (d0, d1, d2)], " n =", len(np.concatenate(d0)))
