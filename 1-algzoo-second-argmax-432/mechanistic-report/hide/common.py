"""Shared helpers for the mechanistic report on the 432-parameter second-argmax RNN (M_16,10).

    h_t = ReLU(w x_t + W h_{t-1}),  h_{-1} = 0,  logits = U h_9       (no biases)

Weights are read (never modified) from the colleague's folder. Everything is float64 NumPy.
Run scripts with `python -B` so no __pycache__ is written anywhere.
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HIDE = Path(__file__).resolve().parent          # scripts, data, notes
OUT = HIDE.parent                               # report.md and figures
ROOT = OUT.parent.parent                        # workspace root
WEIGHTS = (ROOT / "reverse-engineered-networks" / "1-algzoo-second-argmax-432"
           / "code" / "m16_10_weights.npz")

COMP = [1, 6, 7, 8]                             # comparators (large negative input weight)
PULSE = [4, 5, 9]                               # pulse units
WAVE = [0, 2, 3, 10, 11, 12, 13, 14, 15]        # wave / memory units
ORD = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15]   # the 13 non-pulse units
T = 10
N = 16


def load():
    """Return w (16,), W (16,16), U (10,16) as float64."""
    d = np.load(WEIGHTS)
    return (d["W_ih"][:, 0].astype(np.float64), d["W_hh"].astype(np.float64),
            d["W_out"].astype(np.float64))


w, W, U = load()


def run(x, w=w, W=W, return_pre=False, h0=None):
    """Run the RNN on x of shape (B, T') or (T',). Returns h (B, T', 16) [and pre-activations]."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    B, Tn = x.shape
    h = np.zeros((B, N)) if h0 is None else np.broadcast_to(h0, (B, N)).copy()
    H = np.empty((B, Tn, N))
    Z = np.empty((B, Tn, N))
    for t in range(Tn):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        H[:, t], Z[:, t] = h, z
    return (H, Z) if return_pre else H


def logits(H):
    """Logits from the final hidden state. H is (B, T, 16) or (B, 16)."""
    return (H[:, -1] if H.ndim == 3 else H) @ U.T


def sample(n, seed=0, dist="normal", Tn=T):
    rng = np.random.default_rng(seed)
    if dist == "normal":
        return rng.standard_normal((n, Tn))
    if dist == "uniform":
        return rng.uniform(-1, 1, (n, Tn))
    raise ValueError(dist)


def second_argmax(x):
    return np.argsort(x, axis=1)[:, -2]


def running_stats(x):
    """Order statistics of the prefix x_0..x_t (inclusive) for every t.

    Returns dict with arrays (B, T): M1, M2, M3 (largest three values so far, -inf if absent),
    p1, p2, p3 (their positions, -1 if absent), a1, a2, a3 (ages t - p).
    """
    B, Tn = x.shape
    out = {k: np.full((B, Tn), -np.inf) for k in ("M1", "M2", "M3")}
    out.update({k: np.full((B, Tn), -1, dtype=int) for k in ("p1", "p2", "p3")})
    for t in range(Tn):
        pre = x[:, : t + 1]
        order = np.argsort(-pre, axis=1, kind="stable")
        for k in range(min(3, t + 1)):
            idx = order[:, k]
            out[f"M{k+1}"][:, t] = np.take_along_axis(pre, idx[:, None], 1)[:, 0]
            out[f"p{k+1}"][:, t] = idx
    tt = np.arange(Tn)[None, :]
    for k in (1, 2, 3):
        out[f"a{k}"] = np.where(out[f"p{k}"] >= 0, tt - out[f"p{k}"], -1)
    return out


def gates(Z):
    """Boolean gate pattern (pre-activation > 0)."""
    return Z > 0


def pattern_id(G):
    """Pack a (..., 16) boolean gate pattern into one integer per row."""
    return (G.astype(np.int64) * (1 << np.arange(N))).sum(-1)


def pattern_str(pid):
    """Human-readable: list of units that are ON."""
    return "{" + ",".join(str(i) for i in range(N) if (pid >> i) & 1) + "}"


def accuracy(x):
    return (logits(run(x)).argmax(1) == second_argmax(x)).mean()


np.set_printoptions(precision=3, suppress=True, linewidth=200)


def traces(x):
    """Exact input traces for a batch x (B, T): J[b, t, p, :] = d h_t / d x_p with the gates of this
    input held fixed, so that  h_t = sum_p J[b, t, p] * x_p  exactly (the net is piecewise linear
    and bias-free).  Returns (H, J)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    B, Tn = x.shape
    H, Z = run(x, return_pre=True)
    G = (Z > 0).astype(np.float64)                      # (B, T, 16)
    J = np.zeros((B, Tn, Tn, N))
    for t in range(Tn):
        if t > 0:
            J[:, t, :t] = (J[:, t - 1, :t] @ W.T) * G[:, t, None, :]
        J[:, t, t] = G[:, t] * w
    return H, J
