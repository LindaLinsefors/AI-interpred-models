"""Shared helpers for arm H (runner-up events).  Imports the lead's read-only helpers."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
OI = {u: i for i, u in enumerate(ORD)}          # unit -> index inside ORD
K = 10.157                                      # w[4]


class Ev:
    """Everything about a batch, aligned so that index [b, t] describes the arrival of x_t.

    Mo, So, To : largest / 2nd / 3rd largest of x_0..x_{t-1}  (nan at t=0; -inf if absent)
    a1, a2     : t - (position of Mo), t - (position of So)   (age of the old max / old runner-up at step t)
    th[j]      : threshold of unit j at step t, theta_j(t) = -W[j].h_{t-1} / w_j   (j in COMP + [4])
    G          : gates at every step (B, T, 16)
    """

    def __init__(self, x):
        self.x = x
        B = len(x)
        self.H, self.Z = run(x, return_pre=True)
        self.G = self.Z > 0
        self.st = st = running_stats(x)
        self.Hprev = np.concatenate([np.zeros((B, 1, N)), self.H[:, :-1]], 1)
        sh = lambda a, fill: np.concatenate([np.full((B, 1), fill, dtype=a.dtype), a[:, :-1]], 1)
        self.Mo, self.So, self.To = sh(st["M1"], np.nan), sh(st["M2"], np.nan), sh(st["M3"], np.nan)
        tt = np.arange(x.shape[1])[None, :]
        p1o, p2o = sh(st["p1"], -1), sh(st["p2"], -1)
        self.p1o, self.p2o = p1o, p2o
        self.a1 = np.where(p1o >= 0, tt - p1o, -1)
        self.a2 = np.where(p2o >= 0, tt - p2o, -1)
        self.th = {j: -(self.Hprev @ W[j]) / w[j] for j in COMP + [4]}
        self.tt = np.broadcast_to(tt, x.shape)


def lsq(y, cols, intercept=False):
    """Least squares y ~ cols (no intercept by default: the net is homogeneous).
    Returns (coef, R2).  R2 is the centred one (1 - SSE / var(y) / n)."""
    X = np.column_stack(cols + ([np.ones(len(y))] if intercept else []))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return b, 1 - (np.abs(res) ** 2).sum() / (np.abs(y - y.mean()) ** 2).sum()


def q(v, ps=(25, 50, 75)):
    return np.percentile(v, ps)
