"""Verifier's own helpers (verification of the DECODING investigation, 2026-11-23).

Built from the eigen-decomposition of A = W[o, o] directly; nothing is imported from tape_common.py or modes.py
(v0_projector.py compares the two).  Gauge as in modes.py: l_m . W[o, 4] = 1.
    tape coordinates  c = (zeta_alt, Re z_short, Im z_short, Re z_h2, Im z_h2, Re z_h3, Im z_h3)
    rest coordinates  r = (level, runner, Re clock, Im clock, trans, echo, h4, h5, h9)
"""
import sys

ROOT = "c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report"
sys.path.insert(0, ROOT + "/hide")
sys.dont_write_bytecode = True
import numpy as np
from common import w, W, U, run, sample, ORD, second_argmax

A = W[np.ix_(ORD, ORD)]
_lam, _V = np.linalg.eig(A)
_L = np.linalg.inv(_V)
TARGET = {"alt": -0.928, "short": 0.735 * np.exp(1j * np.radians(42.99)), "h2": 0.937 * np.exp(1j * np.radians(83.28)),
          "h3": 0.837 * np.exp(1j * np.radians(124.78)), "level": 1.0099, "runner": 0.9735,
          "clock": 0.951 * np.exp(1j * np.radians(42.75)), "trans": 0.231, "echo": 0.001}
TAPE, CPLX = ("alt", "short", "h2", "h3"), ("short", "h2", "h3", "clock")
LAM, LV, RV = {}, {}, {}
for _n, _t in TARGET.items():
    _k = int(np.argmin(np.abs(_lam - _t)))
    assert abs(_lam[_k] - _t) < 5e-3, (_n, _lam[_k])
    _g = _L[_k] @ W[ORD, 4]
    LAM[_n], LV[_n], RV[_n] = _lam[_k], _L[_k] / _g, _V[:, _k] * _g
assert len({LAM[n] for n in TARGET}) == 9


def mode(H, n):
    z = H[..., ORD] @ LV[n]
    return z if n in CPLX else z.real


def tape_coords(H):
    out = [mode(H, "alt")]
    for n in ("short", "h2", "h3"):
        z = mode(H, n); out += [z.real, z.imag]
    return np.stack(out, -1)


def rest_coords(H):
    zc = mode(H, "clock")
    return np.stack([mode(H, "level"), mode(H, "runner"), zc.real, zc.imag, mode(H, "trans"), mode(H, "echo"),
                     H[..., 4], H[..., 5], H[..., 9]], -1)


def tape13(c):
    """7 tape coordinates -> tape part of h[o] (13 numbers)."""
    out = np.multiply.outer(c[..., 0], RV["alt"].real)
    for j, n in enumerate(("short", "h2", "h3")):
        out = out + 2 * np.real(np.multiply.outer(c[..., 1 + 2 * j] + 1j * c[..., 2 + 2 * j], RV[n]))
    return out


def set_tape(H, c):
    """state with its tape coordinates replaced by c."""
    out = H.copy()
    out[..., ORD] += tape13(c) - tape13(tape_coords(H))
    return out


def advance(c, k, modes=TAPE):
    """the tape coordinates of the listed modes multiplied k times by their eigenvalues."""
    o = c.copy()
    if "alt" in modes: o[:, 0] = c[:, 0] * LAM["alt"].real ** k
    for j, n in enumerate(("short", "h2", "h3")):
        if n in modes:
            z = (c[:, 1 + 2 * j] + 1j * c[:, 2 + 2 * j]) * LAM[n] ** k
            o[:, 1 + 2 * j], o[:, 2 + 2 * j] = z.real, z.imag
    return o


def stats(x, t):
    """top three of x_0..x_t: values M, S, T, positions pM, pS, pT, ages aM, aS, aT, quiet mask."""
    pre = x[:, : t + 1]
    o = np.argsort(-pre, 1, kind="stable")[:, :3]
    v = np.take_along_axis(pre, o, 1)
    d = dict(M=v[:, 0], S=v[:, 1], T=v[:, 2], pM=o[:, 0], pS=o[:, 1], pT=o[:, 2])
    d.update(aM=t - o[:, 0], aS=t - o[:, 1], aT=t - o[:, 2])
    d["quiet"] = (d["aM"] >= 2) & (d["aS"] >= 2) & (d["S"] > 0)
    return d


def answer(h):
    return (h @ U.T).argmax(-1)


def quiet_final(n, seed, dist="normal"):
    """quiet final states of n sequences: x, h, stats (all restricted to the quiet ones)."""
    x = sample(n, seed, dist); h = run(x)[:, 9]; st = stats(x, 9); q = st["quiet"]
    return x[q], h[q], {k: v[q] for k, v in st.items()}


def pooled(n, seed, dist="normal", steps=range(3, 10)):
    """quiet states of all steps pooled: x, H (all sequences), then per state: seq index, step, stats, state."""
    x = sample(n, seed, dist); H = run(x)
    P = {t: stats(x, t) for t in steps}
    rows = [(np.flatnonzero(P[t]["quiet"]), t) for t in steps]
    seq = np.concatenate([r for r, _ in rows]); step = np.concatenate([np.full(len(r), t) for r, t in rows])
    st = {k: np.concatenate([P[t][k][r] for r, t in rows]) for k in P[9] if k != "quiet"}
    return x, H, seq, step, st, H[seq, step]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(v) for v in a)
        print(s, flush=True); self.f.write(s + "\n"); self.f.flush()
