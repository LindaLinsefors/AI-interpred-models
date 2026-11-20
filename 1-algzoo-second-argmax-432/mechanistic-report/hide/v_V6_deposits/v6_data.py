"""Verifier V6: shared data helpers (own code; only common.py / modes.py are imported).

Conventions (claims file): s_t = U h_t; the input of age a (position t - a) sits at index 9 - a.
M_t >= S_t >= T_t are the three largest of x_0..x_t (window includes x_t).
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from pathlib import Path

import numpy as np
from common import *      # w, W, U, run, logits, sample, second_argmax, running_stats, COMP, PULSE, WAVE, ORD
from modes import *       # NAMES, LAM, LEFT, RIGHT, coords

HERE = Path(__file__).resolve().parent
OMEGA = np.radians(42.75)
SEEDS = list(range(5000, 5010))          # 10 batches
NB = 400_000


def top3(x):
    """Own prefix order statistics: values (B, T, 3) and positions (B, T, 3) of the three largest of
    x_0..x_t; -inf / -1 where absent. Written independently of common.running_stats."""
    B, Tn = x.shape
    vals = np.full((B, Tn, 3), -np.inf)
    pos = np.full((B, Tn, 3), -1, dtype=int)
    v = np.full((B, 3), -np.inf)
    p = np.full((B, 3), -1, dtype=int)
    for t in range(Tn):
        xt = x[:, t]
        r0 = xt > v[:, 0]                       # new max
        r1 = ~r0 & (xt > v[:, 1])               # new runner-up
        r2 = ~r0 & ~r1 & (xt > v[:, 2])         # new third
        nv, npos = v.copy(), p.copy()
        # new max: shift everything down
        nv[r0, 1:], npos[r0, 1:] = v[r0, :2], p[r0, :2]
        nv[r0, 0], npos[r0, 0] = xt[r0], t
        nv[r1, 2], npos[r1, 2] = v[r1, 1], p[r1, 1]
        nv[r1, 1], npos[r1, 1] = xt[r1], t
        nv[r2, 2], npos[r2, 2] = xt[r2], t
        v, p = nv, npos
        vals[:, t], pos[:, t] = v, p
    return vals, pos


def batches(seeds=SEEDS, n=NB):
    for sd in seeds:
        x = sample(n, sd)
        H = run(x)
        yield sd, x, H


def score_at(H, t, pos):
    """s_t at the entry holding input position `pos` (array (B,)): index 9 - (t - pos)."""
    s = H[:, t] @ U.T
    return np.take_along_axis(s, (9 - (t - pos))[:, None], 1)[:, 0]


def ols(X, y, intercept=False):
    """Least squares y ~ X b. Returns b, centred R^2 (1 - SSres / SS about mean), uncentred R^2."""
    X = np.column_stack(X) if isinstance(X, (list, tuple)) else X
    if intercept:
        X = np.column_stack([X, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum(), 1 - (r ** 2).sum() / (y ** 2).sum()


if __name__ == "__main__":
    # check own order statistics against common.running_stats and against a brute-force sort
    x = sample(20000, 4999)
    vals, pos = top3(x)
    rs = running_stats(x)
    for k in range(3):
        assert np.array_equal(vals[:, :, k], rs[f"M{k+1}"]), k
        assert np.array_equal(pos[:, :, k], rs[f"p{k+1}"]), k
    for t in range(10):
        srt = np.sort(x[:, : t + 1], axis=1)[:, ::-1]
        for k in range(min(3, t + 1)):
            assert np.array_equal(vals[:, t, k], srt[:, k])
    print("top3 == running_stats == brute-force sort: OK")
    H = run(x)
    print("accuracy on 20k:", (logits(H).argmax(1) == second_argmax(x)).mean())
