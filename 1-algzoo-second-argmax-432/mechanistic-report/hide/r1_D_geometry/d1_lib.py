"""Helpers for the D1 regressions on the Gram matrices of d1_gram.py."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *
from pathlib import Path

HERE = Path(__file__).resolve().parent
COL = dict(g=0, S=1, T=2, x=3, one=17)
YH = list(range(4, 17))          # h_t[ORD]
YP = list(range(18, 21))         # h_t[PULSE]
YF = list(range(21, 34))         # h_t[ORD] with the echo coordinate removed
YZ = list(range(34, 47))         # pre-activation z_t[ORD]


def load_gram(rule="strict"):
    return np.load(HERE / "d1_gram.npz")[rule]            # (t, a1, a2+1, K, K)


def fit(G, regs, ys=YH):
    """Least squares from one Gram matrix G (K, K).  Returns dict with B (len(regs), len(ys)),
    ss_res, ss_y (uncentred), n, mean_res, mean_y  (all per target)."""
    r = [COL[k] for k in regs]
    Grr, Gry = G[np.ix_(r, r)], G[np.ix_(r, ys)]
    B = np.linalg.pinv(Grr, rcond=1e-10) @ Gry
    ss_y = np.diag(G[np.ix_(ys, ys)]).copy()
    ss_res = ss_y - (B * Gry).sum(0)
    n = G[17, 17]
    mean_y = G[17, ys] / n
    mean_res = mean_y - (G[17, r] / n) @ B
    return dict(B=B, ss_res=ss_res, ss_y=ss_y, n=n, mean_res=mean_res, mean_y=mean_y)


def r2(parts):
    """Combine fits of disjoint cells into per-target and overall R^2 (centred and uncentred).
    parts: list of fit() dicts."""
    n = sum(p["n"] for p in parts)
    ss_res = sum(p["ss_res"] for p in parts)
    ss_y = sum(p["ss_y"] for p in parts)
    mres = sum(p["mean_res"] * p["n"] for p in parts) / n
    my = sum(p["mean_y"] * p["n"] for p in parts) / n
    var_res, var_y = ss_res / n - mres ** 2, ss_y / n - my ** 2
    return dict(c=1 - var_res / var_y, u=1 - ss_res / ss_y, c_all=1 - var_res.sum() / var_y.sum(),
                u_all=1 - ss_res.sum() / ss_y.sum(), n=n, rms_res=np.sqrt(ss_res / n), rms_y=np.sqrt(ss_y / n),
                var_res=var_res, var_y=var_y)


def fit_given(G, regs, B, ys=YH):
    """Like fit(), but with a prescribed coefficient matrix B (len(regs), len(ys)) instead of the least-squares one."""
    r = [COL[k] for k in regs]
    Grr, Gry = G[np.ix_(r, r)], G[np.ix_(r, ys)]
    ss_y = np.diag(G[np.ix_(ys, ys)]).copy()
    ss_res = ss_y - 2 * (B * Gry).sum(0) + (B * (Grr @ B)).sum(0)
    n = G[17, 17]
    mean_y = G[17, ys] / n
    mean_res = mean_y - (G[17, r] / n) @ B
    return dict(B=B, ss_res=ss_res, ss_y=ss_y, n=n, mean_res=mean_res, mean_y=mean_y)


def principal_angles(P, Q):
    """Principal angles (deg) between the column spans of P and Q."""
    qp, _ = np.linalg.qr(P)
    qq, _ = np.linalg.qr(Q)
    return np.degrees(np.arccos(np.clip(np.linalg.svd(qp.T @ qq, compute_uv=False), -1, 1)))


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


CLOCK_PLANE = np.stack([RIGHT["clock"].real, RIGHT["clock"].imag], 1)      # (13, 2)
LEVEL = RIGHT["level"].real
RUNNER = RIGHT["runner"].real


def mode_parts(vec, keep):
    """Reconstruct vec (..., 13) keeping only the modes named in keep."""
    m = {}
    for n in NAMES:
        c = vec @ LEFT[n] if n in keep else np.zeros(vec.shape[:-1], complex)
        m[n] = c if IS_COMPLEX[n] else np.real(c)
    return reconstruct(m)
