"""Shared code for the angle "is the four-mode tape a self-contained store?" (t1_selfcontained).

Population: QUIET states (a_M >= 2, a_S >= 2, S_t > 0).  A state at step t is computed as the final state of the
zero-prefixed sequence (0, .., 0, x_0, .., x_t), which is the same state exactly, so the entry of age a is always
at score index 9 - a and only indices 0..7 (ages 2..9) are used.  Entries of the zero prefix are masked out.

The scores are an exact sum of ten PIECES:  s = sum_m U_o (part of h[o] in mode m)  +  U[:, pulse] h[pulse].
PARTS = the nine modes + "pulse".  TAPE = alt, short, h2, h3.  OTHER = level, runner, clock, trans, echo, pulse.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tape_common import *                                   # noqa: E402  (numpy as np, weights, modes, tape helpers)

Uo = U[:, ORD]
PARTS = ["level", "runner", "clock", "trans", "echo", "pulse", "alt", "short", "h2", "h3"]
OTHER = PARTS[:6]
PIECE_MAP = {n: Uo @ projector([n]) for n in NAMES}         # (10, 13): h[o] -> piece of the scores of mode n


def pieces(h):
    """Final states (B, 16) -> dict part -> (B, 10) piece of the scores.  Their sum is U h exactly."""
    out = {n: h[:, ORD] @ PIECE_MAP[n].T for n in NAMES}
    out["pulse"] = h[:, PULSE] @ U[:, PULSE].T
    return out


def population(B, seed, steps=(9,)):
    """Quiet states at the given steps.  Returns a dict of arrays over states:
    xr (right-aligned inputs, zero prefix), h (state), t, M, S, T3, aM, aS, aT (ages), valid (n, 8): entry exists."""
    x = sample(B, seed)
    keep = {k: [] for k in ("xr", "h", "t", "M", "S", "T3", "aM", "aS", "aT")}
    for t in steps:
        st = prefix_stats(x, t); q = st["quiet"]
        xr = np.zeros((q.sum(), 10)); xr[:, 9 - t:] = x[q, : t + 1]
        keep["xr"].append(xr); keep["h"].append(run(xr)[:, 9]); keep["t"].append(np.full(q.sum(), t))
        for k in ("M", "S", "T3", "aM", "aS", "aT"):
            keep[k].append(st[k][q])
    d = {k: np.concatenate(v) for k, v in keep.items()}
    d["valid"] = np.arange(8)[None, :] >= (9 - d["t"])[:, None]         # index i exists iff i >= 9 - t
    d["iS"], d["iM"], d["iT"] = 9 - d["aS"], 9 - d["aM"], 9 - d["aT"]
    return d


def top_entry(A, valid):
    """Index (0..7) of the largest entry of age >= 2 among existing inputs.  A is (n, >= 8)."""
    return np.where(valid, A[:, :8], -np.inf).argmax(1)


def near_precision(A, d, width=0.1):
    """Comparison near the runner-up.  Entries of age >= 2 whose input is within `width` M below S:
    D = (entry - runner-up's entry) / M fitted as b (x - S)/M + c.  Returns slope b, residual sd / b and
    half the interquartile range of the residual / b (both = error in the input, units of M), and n."""
    ar = np.arange(len(A))
    v = (d["xr"][:, :8] - d["S"][:, None]) / d["M"][:, None]
    q = d["valid"] & (v < 0) & (v > -width)
    D = ((A[:, :8] - A[ar, d["iS"]][:, None]) / d["M"][:, None])[q]; v = v[q]
    X = np.stack([v, np.ones(len(v))], 1)
    b = np.linalg.lstsq(X, D, rcond=None)[0]; res = D - X @ b
    iqr = (np.percentile(res, 75) - np.percentile(res, 25)) / 2
    return b[0], res.std() / b[0], iqr / b[0], len(v)


def measures(A, d, top_full):
    """P(runner-up is the top entry), agreement of the top entry with the full score's, slope, input errors."""
    top = top_entry(A, d["valid"])
    b, sd, iq, _ = near_precision(A, d)
    return np.mean(top == d["iS"]), np.mean(top == top_full), b, sd, iq
