"""Verifier's own tape definitions (folder tv_E_dictionary, 2026-11-23).  Only the weights and run/sample come from common.py.

Tape = the eigen-directions of A = W[o, o] (o = 13 non-pulse units) with eigenvalues  -0.928 (alt),  0.735 e^(+-i 42.99 deg)
(short),  0.937 e^(+-i 83.28 deg) (h2),  0.837 e^(+-i 124.78 deg) (h3), found here by matching the eigenvalues.
Gauge of the coordinates: the left vector l_m is scaled so that l_m . W[o, 4] = 1 (a unit pulse of unit 4 writes 1).
zc(H) -> (..., 4) complex coordinates (alt, short, h2, h3);  to7 / from7 <-> 7 real numbers;  set_tape(H, z) replaces them.
Run "python -B v_common.py" for the checks against tape_common.py."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np                                                    # noqa: E402
from common import w, W, U, run, sample, ORD, COMP, PULSE, WAVE       # noqa: E402

A = W[np.ix_(ORD, ORD)]
_lam, _V = np.linalg.eig(A)
_L = np.linalg.inv(_V)
TARGET = np.array([-0.928, 0.735 * np.exp(1j * np.radians(42.99)), 0.937 * np.exp(1j * np.radians(83.28)), 0.837 * np.exp(1j * np.radians(124.78))])
_idx = [int(np.argmin(np.abs(_lam - t))) for t in TARGET]
assert np.abs(_lam[_idx] - TARGET).max() < 2e-3, np.abs(_lam[_idx] - TARGET)
LAMV = _lam[_idx]                                                     # (4,) complex eigenvalues
_c = _L[_idx] @ W[ORD, 4]
LEFTM = _L[_idx] / _c[:, None]                                        # (4, 13), l_m . W[o, 4] = 1
RIGHTM = _V[:, _idx].T * _c[:, None]                                  # (4, 13)
P_TAPE = np.real(np.outer(RIGHTM[0], LEFTM[0])) + sum(2 * np.real(np.outer(RIGHTM[k], LEFTM[k])) for k in (1, 2, 3))
UO = U[:, ORD]


def zc(H):
    z = H[..., ORD] @ LEFTM.T
    z[..., 0] = z[..., 0].real
    return z


def to7(z):
    return np.stack([z[..., 0].real, z[..., 1].real, z[..., 1].imag, z[..., 2].real, z[..., 2].imag, z[..., 3].real, z[..., 3].imag], -1)


def from7(c):
    return np.stack([c[..., 0] + 0j, c[..., 1] + 1j * c[..., 2], c[..., 3] + 1j * c[..., 4], c[..., 5] + 1j * c[..., 6]], -1)


def tape13(z):
    """(..., 4) complex coordinates -> (..., 13) tape part of h[o]."""
    return z[..., 0:1].real * RIGHTM[0].real + 2 * np.real(z[..., 1:] @ RIGHTM[1:])


def set_tape(H, z):
    """State H (..., 16) with its tape coordinates replaced by z (..., 4)."""
    out = H.copy()
    out[..., ORD] += tape13(z) - H[..., ORD] @ P_TAPE.T
    return out


def answer(h):
    return (h @ U.T).argmax(-1)


def top3(x, t):
    """Values, positions and the quiet mask for the prefix x_0..x_t."""
    pre = x[:, : t + 1]
    order = np.argsort(-pre, 1, kind="stable")
    ar = np.arange(len(x))
    pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
    M, S, T3 = pre[ar, pM], pre[ar, pS], pre[ar, pT]
    return dict(M=M, S=S, T3=T3, pM=pM, pS=pS, pT=pT, aM=t - pM, aS=t - pS, aT=t - pT, quiet=(t - pM >= 2) & (t - pS >= 2) & (S > 0))


def prev_top2(x):
    """prevM, prevS (B, 10): maximum and runner-up of x_0..x_(p-1) (-inf where absent)."""
    B, T = x.shape
    pM, pS = np.full((B, T), -np.inf), np.full((B, T), -np.inf)
    m = np.full(B, -np.inf); s = np.full(B, -np.inf)
    for p in range(T):
        pM[:, p], pS[:, p] = m, s
        v = x[:, p]
        s = np.where(v > m, m, np.maximum(s, v)); m = np.maximum(m, v)
    return pM, pS


def r2(c, ch):
    """Pooled R2 over the 7 coordinates (centred on the mean of the real ones)."""
    return 1 - ((c - ch) ** 2).sum() / ((c - c.mean(0)) ** 2).sum()


nrm = lambda D: np.sqrt((np.abs(D) ** 2).sum(-1))
np.set_printoptions(precision=3, suppress=True, linewidth=220)

if __name__ == "__main__":
    import tape_common as tc
    x = sample(20000, 77); h = run(x)[:, 9]
    print("eigenvalues (abs, deg):", np.round(np.abs(LAMV), 4), np.round(np.degrees(np.angle(LAMV)), 2))
    print("projector vs tape_common.P_TAPE:   ", np.abs(P_TAPE - tc.P_TAPE).max(), " rank", np.linalg.matrix_rank(P_TAPE))
    print("coordinates vs tape_common:        ", np.abs(to7(zc(h)) - tc.tape_coords(h)).max())
    print("tape13(zc) vs tape_common.tape_part:", np.abs(tape13(zc(h)) - tc.tape_part(h)[:, ORD]).max())
    z2 = zc(h[::-1].copy())
    print("set_tape vs tape_common.set_tape:  ", np.abs(set_tape(h, z2) - tc.set_tape(h, to7(z2))).max())
    print("set_tape(own coords) = identity:   ", np.abs(set_tape(h, zc(h)) - h).max())
    st = top3(x, 9); ps = tc.prefix_stats(x, 9)
    print("quiet mask equal:", np.array_equal(st["quiet"], ps["quiet"]), " share", st["quiet"].mean())
    print("w on pulse units:", w[PULSE], "  k5 = l.W[o,5]:", np.round(np.abs(LEFTM @ W[ORD, 5]), 3), np.round(np.degrees(np.angle(LEFTM @ W[ORD, 5])), 0))
