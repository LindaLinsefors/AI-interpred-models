"""Verifier's own definitions (independent of tape_common.py and of the investigator's t1_common.py).

Mode projectors are built here from the eigen-decomposition of A = W[o, o] and matched to names by eigenvalue.
A state at step t is taken directly from run(x)[:, t] (no zero prefix); the entry of age a is score index 9 - a.
OLD entries = ages 2..9 = indices 0..7; an entry exists at step t iff its index >= 9 - t.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np                                             # noqa: E402
from common import w, W, U, run, sample, ORD, PULSE, second_argmax   # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=220)
A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
_lam, _V = np.linalg.eig(A)
_L = np.linalg.inv(_V)
_deg = lambda r, a: r * np.exp(1j * np.radians(a))
SPEC = {"level": 1.0099, "runner": 0.9735, "clock": _deg(0.951, 42.75), "h2": _deg(0.937, 83.28), "alt": -0.928,
        "h3": _deg(0.837, 124.78), "short": _deg(0.735, 42.99), "trans": 0.231, "echo": 0.001}
MODES = list(SPEC)
TAPE = ["alt", "short", "h2", "h3"]
OTHER = ["level", "runner", "clock", "trans", "echo", "pulse"]
PROJ, EIG, LVEC, RVEC = {}, {}, {}, {}
for _n, _t in SPEC.items():
    _k = int(np.argmin(np.abs(_lam - _t)))
    assert abs(_lam[_k] - _t) < 5e-3, (_n, _lam[_k])
    _P = np.outer(_V[:, _k], _L[_k])
    PROJ[_n] = (2 * _P.real) if abs(_lam[_k].imag) > 1e-9 else _P.real
    EIG[_n], LVEC[_n], RVEC[_n] = _lam[_k], _L[_k], _V[:, _k]
P_TAPE = sum(PROJ[n] for n in TAPE)
assert np.abs(sum(PROJ.values()) - np.eye(13)).max() < 1e-9


def pieces(h):
    """States (n, 16) -> dict part -> (n, 10) piece of the scores; the ten pieces sum to U h."""
    ho = h[:, ORD]
    out = {n: ho @ (Uo @ PROJ[n]).T for n in MODES}
    out["pulse"] = h[:, PULSE] @ U[:, PULSE].T
    return out


def stats(x, t):
    """Top three of x_0..x_t (values, positions, ages) and the quiet mask a_M >= 2, a_S >= 2, S > 0."""
    pre = x[:, : t + 1]; o = np.argsort(-pre, 1)[:, :3]; ar = np.arange(len(x))
    d = {}
    for k, nm in enumerate(("M", "S", "T")):
        d["p" + nm], d[nm], d["a" + nm] = o[:, k], pre[ar, o[:, k]], t - o[:, k]
    d["quiet"] = (d["aM"] >= 2) & (d["aS"] >= 2) & (d["S"] > 0)
    return d


def population(x, steps=(9,)):
    """Quiet states of the sequences x at the given steps, pooled.  xr = inputs right-aligned (index = 9 - age)."""
    H = run(x); keep = {}
    for t in steps:
        st = stats(x, t); q = st["quiet"]
        xr = np.zeros((q.sum(), 10)); xr[:, 9 - t:] = x[q, : t + 1]
        row = dict(h=H[q, t], xr=xr, t=np.full(q.sum(), t), M=st["M"][q], S=st["S"][q], T=st["T"][q],
                   aM=st["aM"][q], aS=st["aS"][q], aT=st["aT"][q])
        for k, v in row.items():
            keep.setdefault(k, []).append(v)
    d = {k: np.concatenate(v) for k, v in keep.items()}
    d["valid"] = np.arange(8)[None, :] >= (9 - d["t"])[:, None]
    d["iS"], d["iM"], d["iT"] = 9 - d["aS"], 9 - d["aM"], 9 - d["aT"]
    return d


def top_old(Sc, valid):
    """Index of the largest existing old entry (ages 2..9)."""
    return np.where(valid, Sc[:, :8], -np.inf).argmax(1)


def near_slope(Sc, d, width=0.1, extra=None):
    """Old entries whose input is within `width` M below S: fit (entry - runner-up's entry)/M = b (x - S)/M + c.
    Returns b, residual sd / b, half interquartile range of the residual / b, number of entries."""
    ar = np.arange(len(Sc))
    v = (d["xr"][:, :8] - d["S"][:, None]) / d["M"][:, None]
    q = d["valid"] & (v < 0) & (v > -width)
    if extra is not None:
        q = q & extra
    D = ((Sc[:, :8] - Sc[ar, d["iS"]][:, None]) / d["M"][:, None])[q]; v = v[q]
    X = np.stack([v, np.ones(len(v))], 1)
    b = np.linalg.lstsq(X, D, rcond=None)[0]; res = D - X @ b
    return b[0], res.std() / b[0], (np.percentile(res, 75) - np.percentile(res, 25)) / 2 / b[0], len(v)


def cell_fit(Y, F, cell, train):
    """Least squares of Y (n, k) on F (n, f) within each cell, fitted on the rows with train = True; predicts all rows."""
    out = np.zeros_like(Y)
    order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
    for idx in np.split(order, bounds):
        tr = idx[train[idx]]
        if len(tr) > 3 * F.shape[1] + 5:
            out[idx] = F[idx] @ np.linalg.lstsq(F[tr], Y[tr], rcond=None)[0]
    return out


if __name__ == "__main__":
    import tape_common as tc
    print("eigenvalues matched:", {n: f"{abs(EIG[n]):.4f}@{np.degrees(np.angle(EIG[n])):.2f}" for n in MODES})
    print("my P_TAPE vs tape_common.P_TAPE, max abs difference:", np.abs(P_TAPE - tc.P_TAPE).max())
    print("my single-mode projectors vs tape_common.projector:", max(np.abs(PROJ[n] - tc.projector([n])).max() for n in MODES))
    x = sample(50_000, 9001); H = run(x); h = H[:, 9]
    pc = pieces(h); print("pieces sum to U h:", np.abs(sum(pc.values()) - h @ U.T).max())
    print("my tape piece vs tape_common.tape_scores:", np.abs(sum(pc[m] for m in TAPE) - tc.tape_scores(h)).max())
    st = stats(x, 9); st2 = tc.prefix_stats(x, 9)
    print("quiet mask identical:", np.array_equal(st["quiet"], st2["quiet"]), " share", st["quiet"].mean())
    xz = np.zeros_like(x); xz[:, 3:] = x[:, :7]
    print("zero-prefix identity (h_6 vs h_9 of the shifted sequence):", np.abs(run(xz)[:, 9] - H[:, 6]).max())
    print("accuracy:", np.mean((h @ U.T).argmax(1) == second_argmax(x)))
