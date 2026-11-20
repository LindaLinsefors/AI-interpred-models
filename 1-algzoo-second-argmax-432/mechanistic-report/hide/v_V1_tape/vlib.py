"""Small helpers of verifier V1 (own code; only common.py / modes.py are imported from the lead)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import *          # noqa: E402,F401
from modes import *           # noqa: E402,F401


class Tee:
    def __init__(self, name):
        self.f = open(HERE / name, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def top3(x, t):
    """Positions and values of the three largest of x_0..x_t (window includes x_t).
    Returns p (B, 3) int (-1 if absent) and v (B, 3) float (-inf if absent)."""
    pre = x[:, : t + 1]
    k = min(3, t + 1)
    order = np.argsort(-pre, axis=1, kind="stable")[:, :k]
    vals = np.take_along_axis(pre, order, 1)
    p = np.full((x.shape[0], 3), -1)
    v = np.full((x.shape[0], 3), -np.inf)
    p[:, :k], v[:, :k] = order, vals
    return p, v


def scores(H, t):
    """s_t = U h_t restricted to the valid indices, re-indexed by POSITION: column p <-> index 9 - t + p."""
    s = H[:, t] @ U.T
    return s[:, 9 - t:], s
