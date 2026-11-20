"""Reviewer N3: margin of the runner-up over the maximum at t = 9, by distance, gap and age of the maximum.
Run:  python -B n3_margin.py > n3_margin_out.txt
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from common import w, W, U, COMP, sample

np.set_printoptions(precision=1, suppress=True, linewidth=220)
bins = [0.0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]
rows = []
for seed in range(7500, 7515):
    x = sample(400_000, seed)
    h = np.zeros((x.shape[0], 16))
    for t in range(10):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
    L = h @ U.T
    o = np.argsort(-x, axis=1)
    pM, pS = o[:, 0], o[:, 1]
    r = np.arange(len(x))
    M, S = x[r, pM], x[r, pS]
    ok = (S > 0.2) & (pM < 9) & (pS < 9)
    rows.append(np.c_[np.abs(pM - pS), (M - S) / M, (L[r, pS] - L[r, pM]) / M, 9 - pM, (h[:, COMP] > 0).all(1)][ok])
A = np.concatenate(rows)


def table(mask, label):
    tab = np.full((8, 6), np.nan)
    for D in range(1, 9):
        for b in range(6):
            m = mask & (A[:, 0] == D) & (A[:, 1] > bins[b]) & (A[:, 1] <= bins[b + 1])
            if m.sum() > 300:
                tab[D - 1, b] = np.median(A[m, 2])
    print(label)
    print(tab)
    print("   range over cells with gap/M > 0.05: %.1f .. %.1f ; D = 1 only: %.1f .. %.1f" % (
        np.nanmin(tab[:, 1:]), np.nanmax(tab[:, 1:]), np.nanmin(tab[0, 1:]), np.nanmax(tab[0, 1:])))


print("median of (s[pS] - s[pM])/M at t = 9; S > 0.2, both older than x_9; rows D = 1..8, columns gap/M bins", bins)
table(np.ones(len(A), bool), "all ages of the maximum, no condition on the comparators")
table(A[:, 4] == 1, "all ages of the maximum, all four comparators on at t = 9")
table(A[:, 3] >= 2, "maximum at least two steps old")
table(A[:, 3] == 1, "maximum one step old (position 8)")
print("share of selected sequences with the maximum one step old: %.3f" % (A[:, 3] == 1).mean())
