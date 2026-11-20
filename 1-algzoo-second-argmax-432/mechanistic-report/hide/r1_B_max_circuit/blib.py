"""Small shared helpers for arm B (the max circuit). Read-only use of common.py / modes.py."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

W4 = w[4]                                   # 10.157
OI = {u: i for i, u in enumerate(ORD)}      # unit -> index inside ORD

# theta_4(t) = -W[4].h_{t-1}/w_4 = sum_real C[n] m_n + sum_pairs 2 Re(C[n] z_n) + sum_pulse CP[q] h_q
C4 = {n: -(W[4, ORD] @ RIGHT[n]) / W4 for n in NAMES}       # per unit of mode coordinate
CP4 = {q: -W[4, q] / W4 for q in PULSE}                     # per unit of pulse activation


def theta4(Hprev):
    """Threshold of unit 4 for the next input, from the previous state (..., 16)."""
    return -(Hprev @ W[4]) / W4


def theta4_terms(Hprev):
    """Exact split of theta4(Hprev) into mode terms and pulse terms. Returns dict name -> (...,)."""
    m = coords(Hprev)
    out = {}
    for n in NAMES:
        out[n] = 2 * np.real(C4[n] * m[n]) if IS_COMPLEX[n] else np.real(C4[n]) * m[n]
    for q in PULSE:
        out[f"u{q}"] = CP4[q] * Hprev[..., q]
    return out


def pos(a):
    return np.maximum(a, 0.0)


def prefix_stats(x):
    """running_stats plus M1p/M2p (positive parts, 0 if absent) and 'prev' versions shifted by one step
    (value at t refers to the window x_0..x_{t-1}; at t = 0: M = -inf, ages -1)."""
    st = running_stats(x)
    B = x.shape[0]
    out = dict(st)
    for k in ("M1", "M2", "M3"):
        out[k + "prev"] = np.concatenate([np.full((B, 1), -np.inf), st[k][:, :-1]], 1)
    for k in ("a1", "a2", "a3", "p1", "p2", "p3"):
        out[k + "prev"] = np.concatenate([np.full((B, 1), -1), st[k][:, :-1]], 1)
    return out


def slope0(y, X):
    """Least squares through the origin. Returns (coef, uncentred R2, centred R2)."""
    X = np.column_stack(X) if isinstance(X, (list, tuple)) else X.reshape(len(y), -1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return b, 1 - (res ** 2).sum() / (y ** 2).sum(), 1 - res.var() / y.var()


def md_table(header, rows, fmt=None):
    """Markdown table as a string; rows = list of lists."""
    def f(v):
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return str(v)
        return (fmt or "{:.3f}").format(v)
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(f(v) for v in r) + " |\n"
    return s
