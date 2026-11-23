"""Independent re-implementation for the verification of the "sufficiency by resampling" investigation (2026-11-23).

Nothing is imported from the investigator's folder or from tape_common.py / modes.py.  From common.py only the
weights (w, W, U), the unit order ORD and the input sampler are used.

TAPE = the eigen-modes of A = W[o, o] (o = the 13 non-pulse units) with eigenvalues -0.928, 0.735 at +-42.99 deg,
0.937 at +-83.28 deg, 0.837 at +-124.78 deg.  P_TAPE = R[:, S] L[S, :] (S = those 7 eigenvalues, L = R^-1).
Tape coordinates c (7 numbers): l . h[o] with the gauge l . W[o, 4] = 1; real part, then imaginary part.

A STATE is (sequence x, step t); xr = (0, .., 0, x_0, .., x_t) is its right-aligned input (zero-prefix identity),
so the input of age a sits in column 9 - a.  QUIET: a_M >= 2, a_S >= 2, S > 0.

A candidate V is a list of rules; rule(xv) -> (mask of inputs whose age and value belong to V, upper bound that V
implies for the other inputs).  The other inputs are redrawn independently from the input distribution conditioned
on lying below their bound (rejection sampling), and V is re-derived from the new inputs and asserted unchanged.

FU = mean over states of the variance within {original, K redraws} (ddof = 1), summed over coordinates, divided by
the summed variance over all states of the population.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ORD, U, W, sample, w  # noqa: E402

np.seterr(divide="ignore", invalid="ignore")
np.set_printoptions(precision=3, suppress=True, linewidth=220)
o = np.array(ORD)
COL = np.arange(10)

# ------------------------------------------------------------------ modes, built here
lam, R = np.linalg.eig(W[np.ix_(o, o)])
L = np.linalg.inv(R)
KICK = W[o, 4]
SPEC = dict(level=(1.0099, 0), runner=(0.9735, 0), clock=(0.951, 42.75), h2=(0.937, 83.28), alt=(0.928, 180),
            h3=(0.837, 124.78), short=(0.735, 42.99), trans=(0.231, 0), echo=(0.001, 0))
TAPE, REST = ["alt", "short", "h2", "h3"], ["level", "runner", "clock", "trans", "echo"]
IDX = {n: int(np.argmin(np.abs(lam - m * np.exp(1j * np.radians(d))))) for n, (m, d) in SPEC.items()}
assert len(set(IDX.values())) == 9
assert all(abs(lam[IDX[n]] - m * np.exp(1j * np.radians(d))) < 5e-3 for n, (m, d) in SPEC.items()), "spectrum mismatch"


def coord_rows(names):
    """Real matrix whose rows give the mode coordinates (real, then imaginary part for a complex mode)."""
    out = []
    for n in names:
        l = L[IDX[n]] / (L[IDX[n]] @ KICK)
        out += [l.real, l.imag] if abs(lam[IDX[n]].imag) > 1e-9 else [l.real]
    return np.array(out)


def projector(names):
    S = [k for k in range(13) if any(abs(lam[k] - lam[IDX[n]]) < 1e-9 or abs(lam[k] - lam[IDX[n]].conj()) < 1e-9 for n in names)]
    P = R[:, S] @ L[S]
    assert np.abs(P.imag).max() < 1e-9
    return P.real


C_TAPE, C_REST = coord_rows(TAPE), coord_rows(REST)          # (7, 13), (6, 13)
P_TAPE = projector(TAPE)
G_TAPE = U[:, o] @ P_TAPE                                    # tape piece of the scores = G_TAPE h[o]
COORD = ["alt", "sh.re", "sh.im", "h2.re", "h2.im", "h3.re", "h3.im"]


def fwd(x, h0=None):
    """State after feeding the columns of x (B, T') one by one, starting from h0 (zero if None)."""
    h = np.zeros((len(x), 16)) if h0 is None else h0
    for j in range(x.shape[1]):
        h = np.maximum(np.outer(x[:, j], w) + h @ W.T, 0.0)
    return h


def tape_part(h):
    out = np.zeros_like(h)
    out[:, o] = h[:, o] @ P_TAPE.T
    return out


def finish(h, x, t):
    """The network's answer when the run goes on from state h (reached at step t) with the original later inputs."""
    out = np.empty(len(h), int)
    for tt in np.unique(t):
        q = t == tt
        out[q] = (fwd(x[q, tt + 1:], h[q]) @ U.T).argmax(1)
    return out


def feats(h, M):
    """Per-state quantities in units of M: tape coordinates (7), rest mode coordinates (6), pulse units (3),
    tape piece of the scores at ages >= 2 (8), and tape piece / rest piece / full score with the mean over the ten
    entries removed, ages >= 2 (8 each)."""
    ho = h[:, o] / M[:, None]
    sT, sF = ho @ G_TAPE.T, (h / M[:, None]) @ U.T
    cen = lambda s: (s - s.mean(1, keepdims=True))[:, :8]
    return dict(cT=ho @ C_TAPE.T, cR=ho @ C_REST.T, pulse=h[:, [4, 5, 9]] / M[:, None], gT=sT[:, :8],
                dT=cen(sT), dR=cen(sF - sT), dF=cen(sF))


# ------------------------------------------------------------------ population
class Pop:
    """All quiet states at the given steps of n_seq natural sequences (no other filter)."""

    def __init__(self, n_seq=40_000, seed=11, steps=range(3, 10), dist="normal"):
        x0 = sample(n_seq, seed, dist)
        ar0 = np.arange(n_seq)
        X, T = [], []
        for t in steps:
            srt = np.argsort(x0[:, :t + 1], 1)
            pM, pS = srt[:, -1], srt[:, -2]
            q = (t - pM >= 2) & (t - pS >= 2) & (x0[ar0, pS] > 0)
            X.append(x0[q]); T.append(np.full(q.sum(), t))
        self.x, self.t, self.dist = np.concatenate(X), np.concatenate(T), dist
        self.n = len(self.t)
        self.real = COL >= (9 - self.t)[:, None]
        self.xr = np.zeros((self.n, 10))
        for t in steps:
            q = self.t == t
            self.xr[q, 9 - t:] = self.x[q, :t + 1]
        self.xv = np.where(self.real, self.xr, -np.inf)
        srt = np.argsort(self.xv, 1)
        ar = np.arange(self.n)
        self.cM, self.cS = srt[:, -1], srt[:, -2]                         # columns of maximum and runner-up
        self.M, self.S = self.xr[ar, self.cM], self.xr[ar, self.cS]
        self.aM, self.aS = 9 - self.cM, 9 - self.cS
        self.h = fwd(self.xr)
        self.f = feats(self.h, self.M)
        self.ans, self.truth = finish(self.h, self.x, self.t), np.argsort(self.x, 1)[:, -2]


# ------------------------------------------------------------------ rules: xv (N, 10), -inf outside the real columns
def top(k):
    """The k largest inputs so far."""
    def f(xv):
        thr = np.sort(xv, 1)[:, -k][:, None]
        return xv >= thr, np.broadcast_to(thr, xv.shape)
    return f


def contenders(c):
    """Inputs that on arrival exceeded c x max(second largest of the earlier inputs, 0)."""
    def f(xv):
        a, b = np.zeros(len(xv)), np.zeros(len(xv))                      # largest, second largest so far (floored at 0)
        bound = np.empty_like(xv)
        for j in range(10):
            x = xv[:, j]
            bound[:, j] = c * b
            a, b = np.maximum(a, x), np.maximum(b, np.minimum(a, x))
        return xv > bound, bound
    return f


def arrival(rank, floor=0.0, c=1.0, ref=None):
    """Inputs that on arrival exceeded c x (the rank-th largest of the earlier inputs and of `rank` virtual inputs
    equal to floor).  ref: use this rank for the threshold instead (ref = 1: the running maximum)."""
    def f(xv):
        lead = np.full((len(xv), rank), floor)                           # ascending
        bound = np.empty_like(xv)
        for j in range(10):
            bound[:, j] = c * lead[:, 0 if ref is None else rank - ref]
            lead = np.sort(np.column_stack([lead, xv[:, j]]), 1)[:, 1:]
        return xv > bound, bound
    return f


def above_final(c):
    """Inputs above c x the runner-up S_t of the state."""
    def f(xv):
        b = c * np.sort(xv, 1)[:, -2][:, None]
        return xv > b, np.broadcast_to(b, xv.shape)
    return f


def ages(a):
    def f(xv):
        m = np.zeros(xv.shape, bool)
        m[:, [9 - k for k in a]] = True
        return m, np.full(xv.shape, np.inf)
    return f


def segment(which, k=10):
    """The k largest inputs of a segment: old = older than both maximum and runner-up, mid = between them,
    new = younger than both, pre = old + mid (everything that arrived before the younger of the two)."""
    def f(xv):
        two = np.argsort(xv, 1)[:, -2:]
        lo, hi = two.min(1)[:, None], two.max(1)[:, None]
        seg = dict(old=COL < lo, mid=(COL > lo) & (COL < hi), new=COL > hi, pre=(COL < hi) & (COL != lo))[which] & np.isfinite(xv)
        thr = np.sort(np.where(seg, xv, -np.inf), 1)[:, -k][:, None] if k < 10 else np.full((len(xv), 1), -np.inf)
        return seg & (xv >= thr), np.where(seg, thr, np.inf)
    return f


def fixed(P, xv, rules):
    mask, bound = np.zeros(xv.shape, bool), np.full(xv.shape, np.inf)
    for r in rules:
        m, b = r(xv)
        mask, bound = mask | m, np.minimum(bound, b)
    return mask & P.real, bound


def draw_below(bound, need, rng, dist="normal"):
    """Independent draws from the input distribution conditioned on < bound, at the places where need is True."""
    prop = (lambda n: rng.standard_normal(n)) if dist == "normal" else (lambda n: rng.uniform(-1, 1, n))
    idx = np.flatnonzero(need.ravel())
    b = bound.ravel()[idx]
    z = prop(len(idx))
    for _ in range(80):
        bad = z >= b
        if not bad.any():
            break
        z[bad] = prop(int(bad.sum()))
    bad = z >= b
    if bad.any():                                                         # far tail: inverse CDF
        u = rng.uniform(1e-300, 1, int(bad.sum()))
        z[bad] = ndtri(u * ndtr(b[bad])) if dist == "normal" else -1 + u * (np.minimum(b[bad], 1) + 1)
    out = np.zeros(bound.size)
    out[idx] = z
    return out.reshape(bound.shape)


def resample(P, rules, rng):
    """New right-aligned inputs with V held fixed exactly; returns (inputs, mask of the copied inputs)."""
    mask, bound = fixed(P, P.xv, rules)
    free = P.real & ~mask
    new = np.where(free, draw_below(bound, free, rng, P.dist), P.xr)
    m2, b2 = fixed(P, np.where(P.real, new, -np.inf), rules)
    assert (m2 == mask).all() and np.allclose(b2[free], bound[free]), "V changed under resampling"
    return new, mask


def V(P, *rules):
    """Sampler for V = step + maximum + runner-up + the given rules."""
    return lambda rng: resample(P, (top(2),) + rules, rng)


def unrelated(P):
    return lambda rng: (P.xr[rng.permutation(P.n)], None)


# ------------------------------------------------------------------ evaluation
def evaluate(P, sampler, K=3, seed=0, transplant=True):
    """Per-state within-V variances of every feature group, and what the three transplants do to the answer:
    tape = recipient keeps its rest and gets the donor's tape part; rest = the reverse; whole = the donor's state."""
    rng = np.random.default_rng(seed)
    F = {k: [v] for k, v in P.f.items()}
    cnt = {k: np.zeros(P.n) for k in ("tape", "rest", "whole", "acc", "clip", "neg", "kept")}
    tp_own = tape_part(P.h)
    for _ in range(K):
        xk, mask = sampler(rng)
        hk = fwd(xk)
        Mk = xk.max(1)
        for k, v in feats(hk, Mk).items():
            F[k].append(v)
        cnt["kept"] += 0 if mask is None else mask.sum(1)
        if not transplant:
            continue
        hk = hk * (P.M / Mk)[:, None]                                     # donor in the recipient's units
        tp_don = tape_part(hk)
        hyb = P.h - tp_own + tp_don
        a = finish(hyb, P.x, P.t)
        cnt["tape"] += a != P.ans
        cnt["acc"] += a == P.truth
        cnt["clip"] += finish(np.maximum(hyb, 0), P.x, P.t) != P.ans
        cnt["neg"] += (hyb < -1e-9).any(1)
        cnt["rest"] += finish(hk - tp_don + tp_own, P.x, P.t) != P.ans
        cnt["whole"] += finish(hk, P.x, P.t) != P.ans
    out = {k: np.var(v, 0, ddof=1) for k, v in F.items()}
    out.update({k: v / K for k, v in cnt.items()})
    return out


def fu(P, r, key="cT", q=None, per_coord=False):
    q = np.ones(P.n, bool) if q is None else q
    num, den = r[key][q].mean(0), P.f[key].var(0)
    return num / den if per_coord else num.sum() / den.sum()


HEAD = (f"{'V':44s} {'kept':>5s} | " + " ".join(f"{c:>6s}" for c in COORD) + " | FUtape FUscor FUrest | answer changed %: tape   rest  whole |"
        "  acc%  | clip   neg%")


def line(P, name, r, q=None):
    q = np.ones(P.n, bool) if q is None else q
    pc = lambda k: 100 * r[k][q].mean()
    return (f"{name:44s} {r['kept'][q].mean():5.2f} | " + " ".join(f"{v:6.3f}" for v in fu(P, r, "cT", q, True))
            + f" | {fu(P, r, 'cT', q):6.3f} {fu(P, r, 'gT', q):6.3f} {fu(P, r, 'cR', q):6.3f} |"
            + f"            {pc('tape'):6.2f} {pc('rest'):6.2f} {pc('whole'):6.2f} | {pc('acc'):6.2f} | {pc('clip'):5.2f} {pc('neg'):5.1f}")
