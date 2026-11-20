"""Verifier V4 (core): shared setup. Own definitions, built only from common.py / modes.py and the claims file."""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from pathlib import Path

from common import *   # noqa: F401,F403
from modes import *    # noqa: F401,F403

VDIR = Path(__file__).resolve().parent
CORE = [2, 4, 1, 6, 7, 8]                       # order used in claim K5
OTHER10 = [u for u in range(16) if u not in CORE]
WAVE_NO2 = [u for u in WAVE if u != 2]

g = -w[COMP]                                    # (4,) gauge of the comparators, g_j = -w_j
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
b2 = W[COMP, 2] / g
b4 = W[COMP, 4] / g


def thresholds(H):
    """theta_j(t) = (W h_{t-1})_j / g_j for j in COMP, t = 0..T (index T = 'theta(T)', after the last state).
    Returns (B, T+1, 4); also theta4 (B, T+1) = -(W h_{t-1})_4 / w_4."""
    B, Tn, _ = H.shape
    Hprev = np.concatenate([np.zeros((B, 1, 16)), H], axis=1)        # index t -> h_{t-1}
    th = (Hprev @ W[COMP].T) / g
    th4 = -(Hprev @ W[4]) / w[4]
    return th, th4


class Tee:
    def __init__(self, name):
        self.f = open(VDIR / name, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()
