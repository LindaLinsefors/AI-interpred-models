"""Verifier V3 (interventions): own helpers, written from the definitions in claims_for_verification.md.

    h_t = ReLU(w x_t + W h_{t-1}),  h_{-1} = 0,  logits = U h_9
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *   # noqa: E402,F401
from modes import *    # noqa: E402,F401

HERE = Path(__file__).resolve().parent
OMEGA = float(np.angle(LAM["clock"]))          # 42.75 deg in radians
ORD_A = np.array(ORD)
COMP_A = np.array(COMP)


def run_iv(x, tint, fn, return_state=False):
    """Run the net on x (B, 10). For row b, right after h_{tint[b]} is computed, replace it by fn(h, rows).

    fn gets the (n, 16) states of the rows intervened at this step and their row indices; it must
    return the new (n, 16) states. Returns the final state h_9 (B, 16) [and the intervened states]."""
    x = np.asarray(x, dtype=np.float64)
    B = x.shape[0]
    h = np.zeros((B, N))
    hiv = np.zeros((B, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
        rows = np.flatnonzero(tint == t)
        if rows.size:
            h[rows] = fn(h[rows], rows)
            hiv[rows] = h[rows]
    return (h, hiv) if return_state else h


def answer(h9):
    return (h9 @ U.T).argmax(1)


def scale_units(units, c=0.0):
    units = np.array(units)

    def fn(h, rows):
        h = h.copy()
        h[:, units] *= c
        return h
    return fn


def mode_fn(mode="clock", f=1.0, k=0.0, angle=None, clip=False, units=None):
    """z' = f z exp(i k angle) for a complex mode (angle defaults to the mode's own angle);
    for a real mode z' = f z.  h[ORD] += 2 Re((z' - z) RIGHT)  (no factor 2 for a real mode).
    units: if given, apply the change only to these units (others left untouched).
    clip: clip the intervened state at zero afterwards."""
    L, R = LEFT[mode], RIGHT[mode]
    ang = float(np.angle(LAM[mode])) if angle is None else angle

    def fn(h, rows):
        h = h.copy()
        z = h[:, ORD_A] @ L
        if IS_COMPLEX[mode]:
            dz = z * (f * np.exp(1j * k * ang) - 1.0)
            d = 2 * np.real(np.multiply.outer(dz, R))
        else:
            dz = np.real(z) * (f - 1.0)
            d = np.real(np.multiply.outer(dz, R))
        if units is not None:
            keep = np.isin(ORD_A, units)
            d = d * keep
        h[:, ORD_A] += d
        if clip:
            h = np.maximum(h, 0.0)
        return h
    return fn


def select_I1(x):
    """I1 selection: second largest > 0.2 and largest - second > 0.2. Returns ok, pM, pS, M, S."""
    order = np.argsort(x, axis=1)
    pM, pS = order[:, -1], order[:, -2]
    r = np.arange(x.shape[0])
    M, S = x[r, pM], x[r, pS]
    return (S > 0.2) & (M - S > 0.2), pM, pS, M, S


def select_I2(x):
    """I2 selection. Returns ok, tr, p0, q0, M0, S0 (prefix x_0..x_{tr-1} statistics)."""
    B = x.shape[0]
    r = np.arange(B)
    order = np.argsort(x, axis=1)
    tr = order[:, -1]
    xp = np.where(np.arange(T)[None, :] < tr[:, None], x, -np.inf)
    o2 = np.argsort(xp, axis=1)
    p0, q0 = o2[:, -1], o2[:, -2]
    M0, S0 = xp[r, p0], xp[r, q0]
    ok = ((tr >= 3) & (tr <= 8) & (tr - 1 - p0 >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1)
          & (order[:, -2] == p0) & (order[:, -3] == q0))
    return ok, tr, p0, q0, M0, S0


def gather_I2(n_target, seed0, batch=400_000, max_batches=200):
    """Collect I2-selected sequences from fresh batches (seeds seed0, seed0+1, ...)."""
    xs, trs, p0s, q0s, M0s, S0s = [], [], [], [], [], []
    n, nb = 0, 0
    while n < n_target and nb < max_batches:
        x = sample(batch, seed0 + nb)
        ok, tr, p0, q0, M0, S0 = select_I2(x)
        xs.append(x[ok]); trs.append(tr[ok]); p0s.append(p0[ok]); q0s.append(q0[ok])
        M0s.append(M0[ok]); S0s.append(S0[ok])
        n += ok.sum(); nb += 1
    cat = np.concatenate
    return cat(xs), cat(trs), cat(p0s), cat(q0s), cat(M0s), cat(S0s), nb


def gather_I1(n_target, seed0, batch=400_000, max_batches=200):
    xs, pMs, pSs, Ms, Ss = [], [], [], [], []
    n, nb = 0, 0
    while n < n_target and nb < max_batches:
        x = sample(batch, seed0 + nb)
        ok, pM, pS, M, S = select_I1(x)
        xs.append(x[ok]); pMs.append(pM[ok]); pSs.append(pS[ok]); Ms.append(M[ok]); Ss.append(S[ok])
        n += ok.sum(); nb += 1
    cat = np.concatenate
    return cat(xs), cat(pMs), cat(pSs), cat(Ms), cat(Ss), nb


class Tee:
    """print to stdout and to a text file in this folder."""

    def __init__(self, name):
        self.f = open(HERE / name, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(v) for v in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
