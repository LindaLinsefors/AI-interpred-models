"""Shared helpers for arm F (anatomy of a record)."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *          # noqa
from modes import *           # noqa

np.set_printoptions(precision=3, suppress=True, linewidth=220)
W4 = 10.157                   # w[4], rounded; used only for normalising printed numbers
A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
OI = {u: i for i, u in enumerate(ORD)}          # unit -> index inside ORD
CI = [OI[u] for u in COMP]
WI = [OI[u] for u in WAVE]


def records(x, s, st=None):
    """Boolean mask of sequences with a record (strict new max) at step s >= 2 whose old max is > 0,
    plus a dict of per-sequence quantities:
      Mn = x_s, M0 = old max, p0 = its position, g = s - p0 (age of the old max AT the record step),
      S0 = old runner-up (second largest of x_0..x_{s-1}), S0p = max(S0, 0), q0 = position of S0.
    """
    st = st or running_stats(x)
    M0 = st["M1"][:, s - 1]
    rec = (x[:, s] > M0) & (M0 > 0)
    S0 = st["M2"][:, s - 1]
    return rec, dict(Mn=x[:, s], M0=M0, p0=st["p1"][:, s - 1], g=s - st["p1"][:, s - 1],
                     S0=S0, S0p=np.maximum(S0, 0), q0=st["p2"][:, s - 1])


def ols(y, cols, intercept=False):
    """Least squares y ~ cols (no intercept by default: the net is homogeneous).
    Returns (beta, R2) with R2 = 1 - SSres / SStot (centred)."""
    X = np.column_stack(cols + ([np.ones(len(y))] if intercept else []))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()


def cols_ols(y, cols):
    """Complex least squares (no intercept); uncentred R2."""
    X = np.column_stack(cols).astype(complex)
    b, *_ = np.linalg.lstsq(X, y.astype(complex), rcond=None)
    r = y - X @ b
    return b, 1 - (np.abs(r) ** 2).sum() / (np.abs(y) ** 2).sum()


def fmtc(c):
    return f"{abs(c):6.3f}@{np.degrees(np.angle(c)):6.1f}"


def on_str(g):
    """g: boolean (16,) -> '{units on}'"""
    return "{" + ",".join(str(i) for i in range(N) if g[i]) + "}"


def final_jac(x):
    """Exact Jacobian of the final state: J9[b, p, :] = d h_9 / d x_p (gates of this input held fixed),
    so h_9 = sum_p J9[b, p] x_p.  Returns H (B,T,16), Z (B,T,16), J9 (B,T,16)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    B, Tn = x.shape
    H, Z = run(x, return_pre=True)
    Gt = (Z > 0).astype(np.float64)
    J = np.zeros((B, Tn, N))
    for t in range(Tn):
        if t > 0:
            J[:, :t] = (J[:, :t] @ W.T) * Gt[:, t, None, :]
        J[:, t] = Gt[:, t] * w
    return H, Z, J


def run_from(h, x, t0, clamp=None):
    """Continue the RNN from state h (B,16) = h_{t0-1} with inputs x[:, t0:], returns H (B, T-t0, 16), Z.
    clamp: optional function (t, z) -> h applied instead of ReLU."""
    B = x.shape[0]
    Hs, Zs = [], []
    for t in range(t0, x.shape[1]):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0) if clamp is None else clamp(t, z)
        Hs.append(h); Zs.append(z)
    return np.stack(Hs, 1), np.stack(Zs, 1)
