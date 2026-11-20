"""Verifier V1: T3 at finer resolution (seeds 6101..6105, 200k each): accuracy given S_t > 0
by (age of max, age of runner-up) at t = 9, and by separation |p1 - p2| pooled over t.
Run: python -B v1_t3_cells.py -> v1_t3_cells.txt
"""
from vlib import *

P = Tee("v1_t3_cells.txt")
n_c = np.zeros((10, 10)); c_c = np.zeros((10, 10))
n_s = np.zeros((10, 10)); c_s = np.zeros((10, 10))          # [t, separation]
n_3 = np.zeros((10, 10)); c_3 = np.zeros((10, 10))          # [t, |p2 - p3|]
for seed in range(6101, 6106):
    x = sample(200_000, seed)
    H = run(x)
    for t in range(1, 10):
        sub, _ = scores(H, t)
        p, v = top3(x, t)
        ok = sub.argmax(1) == p[:, 1]
        pos = v[:, 1] > 0
        sep = np.abs(p[:, 0] - p[:, 1])
        np.add.at(n_s[t], sep[pos], 1); np.add.at(c_s[t], sep[pos], ok[pos])
        if t >= 2:
            s3 = np.abs(p[:, 1] - p[:, 2])
            np.add.at(n_3[t], s3[pos], 1); np.add.at(c_3[t], s3[pos], ok[pos])
        if t == 9:
            np.add.at(n_c, (9 - p[pos, 0], 9 - p[pos, 1]), 1); np.add.at(c_c, (9 - p[pos, 0], 9 - p[pos, 1]), ok[pos])
    del H
P("=== t = 9, S_9 > 0: accuracy by age of the max a1 (rows 0..9) and age of the runner-up a2 (columns 0..9) ===")
with np.errstate(invalid="ignore", divide="ignore"):
    acc = c_c / n_c
for a in range(10):
    P(f" a1={a}: " + " ".join("  .  " if a == b else f"{acc[a, b]:.3f}" for b in range(10)))
off = ~np.eye(10, dtype=bool)
P(f" min cell {np.nanmin(acc[off]):.3f} at (a1,a2)={np.unravel_index(np.nanargmin(np.where(off, acc, np.nan)), acc.shape)}; max cell {np.nanmax(acc[off]):.3f}; cells below 0.93: {int((acc[off] < 0.93).sum())} of 90; n per cell about {int(n_c[off].mean())}")
P("\n=== accuracy given S_t > 0 by separation |p1 - p2| (columns 1..t) ===")
for t in range(1, 10):
    P(f" t={t}: " + " ".join(f"{c_s[t, s]/n_s[t, s]:.3f}" for s in range(1, t + 1)))
P(" pooled over t: " + " ".join(f"{c_s[:, s].sum()/n_s[:, s].sum():.3f}" for s in range(1, 10)))
P("\n=== accuracy given S_t > 0 by separation between runner-up and third |p2 - p3| (columns 1..t) ===")
for t in range(2, 10):
    P(f" t={t}: " + " ".join(f"{c_3[t, s]/n_3[t, s]:.3f}" for s in range(1, t + 1)))
P(" pooled over t: " + " ".join(f"{c_3[:, s].sum()/n_3[:, s].sum():.3f}" for s in range(1, 10)))
