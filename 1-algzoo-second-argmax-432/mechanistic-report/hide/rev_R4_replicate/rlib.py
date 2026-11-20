"""Reviewer R4: own helpers for the spot replication (independent of the r1_* arm scripts).

    h_t = ReLU(w x_t + W h_{t-1}),  logits = U h_9
Only the weights and the basic helpers of hide/common.py and hide/modes.py are used.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")

import numpy as np
from common import w, W, U, COMP, PULSE, WAVE, ORD

np.set_printoptions(precision=4, suppress=True, linewidth=220)
G = -w[COMP]            # g_j = -w_j for the comparators 1, 6, 7, 8
W4 = w[4]


def fwd(x, hook=None):
    """Forward pass with an optional hook(t, z, h) -> h that may overwrite activations.
    Returns H (B, T, 16) post-hook activations and Z (B, T, 16) pre-activations."""
    B, T = x.shape
    h = np.zeros((B, 16))
    H = np.empty((B, T, 16))
    Z = np.empty((B, T, 16))
    for t in range(T):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        if hook is not None:
            h = hook(t, z, h)
        H[:, t], Z[:, t] = h, z
    return H, Z


def prefix_stats(x):
    """Running top three of x_0..x_t (inclusive) for every t.
    Returns dict of (B, T) arrays: M, S, T3 (values, -inf if absent), pM, pS, pT (positions, -1 if absent)."""
    B, T = x.shape
    out = {k: np.full((B, T), -np.inf) for k in ("M", "S", "T3")}
    out.update({k: np.full((B, T), -1, dtype=np.int64) for k in ("pM", "pS", "pT")})
    m = np.full(B, -np.inf); s = m.copy(); th = m.copy()
    pm = np.full(B, -1, dtype=np.int64); ps = pm.copy(); pth = pm.copy()
    for t in range(T):
        v = x[:, t]
        nm = v > m
        ns = ~nm & (v > s)
        nt = ~nm & ~ns & (v > th)
        th, pth = np.where(nm | ns, s, np.where(nt, v, th)), np.where(nm | ns, ps, np.where(nt, t, pth))
        s, ps = np.where(nm, m, np.where(ns, v, s)), np.where(nm, pm, np.where(ns, t, ps))
        m, pm = np.where(nm, v, m), np.where(nm, t, pm)
        for k, a in (("M", m), ("S", s), ("T3", th), ("pM", pm), ("pS", ps), ("pT", pth)):
            out[k][:, t] = a
    return out


def prev(a, fill):
    """Shift a (B, T) array of inclusive prefix stats to 'before x_t' (value at t-1; fill at t=0)."""
    b = np.empty_like(a)
    b[:, 0] = fill
    b[:, 1:] = a[:, :-1]
    return b


def lsq(X, y):
    """Least squares without intercept. Returns coef, centred R2, uncentred R2."""
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    return c, 1 - r.var() / y.var(), 1 - (r ** 2).sum() / (y ** 2).sum()


def gauss(n, seed):
    return np.random.default_rng(seed).standard_normal((n, 10))
