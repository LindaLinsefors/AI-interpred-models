"""Resampling tools for "what does the tape store?"  (angle: sufficiency by resampling, 2026-11-23).

A STATE is (sequence x, step t).  By the zero-prefix identity it is the final state of the right-aligned input
xr = (0, .., 0, x_0, .., x_t), so the input of age a sits at column 9 - a of xr.  "real" marks the columns 9-t..9.

A candidate V is a list of RULES.  A rule maps the inputs to (mask, bound):  mask = the inputs whose age and value
belong to V (they are copied), bound = the upper limit that V implies for every other input at that column.
Given V, the other inputs are independent N(0,1) truncated below their bound, so resample() draws from the exact
conditional distribution; it re-derives mask and bound from the new inputs and asserts that V is unchanged.

FU(V) = E[Var(c / M | V)] / Var(c / M): the within-group variance is taken over the original state and its K
resamples (K + 1 exchangeable draws given V, ddof = 1), the denominator over the original quiet states.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scipy.special import ndtr, ndtri

from tape_common import *

np.seterr(divide="ignore", invalid="ignore")
COORD = ["alt", "short.re", "short.im", "h2.re", "h2.im", "h3.re", "h3.im"]
COL = np.arange(10)


# ------------------------------------------------------------------ population
class Pop:
    """All quiet states at steps t = 3..9 of B natural N(0,1) sequences (no other filter)."""

    def __init__(self, B=80_000, seed=0, steps=range(3, 10)):
        x0 = sample(B, seed)
        keep = [(x0[prefix_stats(x0, t)["quiet"]], t) for t in steps]
        self.x = np.concatenate([k[0] for k in keep])                     # full original sequences
        self.t = np.concatenate([np.full(len(k[0]), k[1]) for k in keep])
        self.n = len(self.t)
        self.xr = np.zeros_like(self.x)
        for t in steps:
            q = self.t == t
            self.xr[q, 9 - t:] = self.x[q, :t + 1]
        self.real = COL[None, :] >= (9 - self.t)[:, None]
        self.h = run(self.xr)[:, 9]
        st = stats(self.xr, self.real)
        self.M, self.S, self.aM, self.aS = st["M"], st["S"], st["aM"], st["aS"]
        self.c = tape_coords(self.h) / self.M[:, None]                    # tape coordinates in units of M
        self.g = tape_scores(self.h)[:, :8] / self.M[:, None]             # tape piece of the scores, ages 9..2
        self.ans = finish(self.h, self.x, self.t)                          # the network's answer, unswapped
        self.truth = second_argmax(self.x)


def stats(xr, real):
    xv = np.where(real, xr, -np.inf)
    order = np.argsort(-xv, 1, kind="stable")
    ar = np.arange(len(xr))
    return dict(M=xv[ar, order[:, 0]], S=xv[ar, order[:, 1]], aM=9 - order[:, 0], aS=9 - order[:, 1], order=order)


def finish(h, x, t):
    """The network's answer when the run goes on from state h (reached at step t) with the original later inputs."""
    out = np.empty(len(h), int)
    for tt in np.unique(t):
        q = t == tt
        out[q] = answer(h[q] if tt == 9 else continue_from(h[q], x[q, tt + 1:])[:, -1])
    return out


# ------------------------------------------------------------------ rules: xv (N, 10), -inf where not real
def r_top(k):
    """The k largest inputs so far (k = 2: maximum and runner-up)."""
    def f(xv):
        b = -np.sort(-xv, 1)[:, k - 1:k]
        return xv >= b, np.broadcast_to(b, xv.shape)
    return f


def r_above(c):
    """Every input above c times the runner-up S_t."""
    def f(xv):
        b = c * -np.sort(-xv, 1)[:, 1:2]
        return xv > b, np.broadcast_to(b, xv.shape)
    return f


def r_records(k, floor=0.0):
    """Every input that was among the k largest when it arrived, i.e. exceeded the k-th largest of the earlier
    inputs (and of k virtual earlier inputs equal to floor; floor = -inf gives the plain definition)."""
    def f(xv):
        top = np.full((len(xv), k), floor)
        b = np.empty_like(xv)
        for j in range(10):
            b[:, j] = top[:, 0]
            top = np.sort(np.concatenate([top, xv[:, j:j + 1]], 1), 1)[:, 1:]
        return xv > b, b
    return f


def r_runthr(c):
    """Every input above c times the runner-up of the inputs before it (floored at 0)."""
    def f(xv):
        b = c * r_records(2)(xv)[1]
        return xv > b, b
    return f


def r_ages(ages):
    """The inputs of the given ages, whatever their values."""
    def f(xv):
        m = np.zeros(xv.shape, bool)
        m[:, [9 - a for a in ages]] = True
        return m, np.full(xv.shape, np.inf)
    return f


def fixed(xv, rules):
    mask, bound = np.zeros(xv.shape, bool), np.full(xv.shape, np.inf)
    for r in rules:
        m, b = r(xv)
        mask, bound = mask | m, np.minimum(bound, b)
    return mask & np.isfinite(xv), bound


def resample(xr, real, rules, rng):
    """New inputs with V (the rules) held fixed exactly.  Returns (new xr, mask of copied inputs)."""
    mask, bound = fixed(np.where(real, xr, -np.inf), rules)
    free = real & ~mask
    new = np.where(free, ndtri(rng.uniform(1e-300, 1, xr.shape) * ndtr(bound)), xr)
    m2, b2 = fixed(np.where(real, new, -np.inf), rules)
    assert (m2 == mask).all() and np.allclose(b2[free], bound[free]), "V changed under resampling"
    return new, mask


# ------------------------------------------------------------------ evaluation
def evaluate(P, sampler, K=4, seed=0, part="tape", swap=True, extra=False):
    """sampler(rng) -> (new right-aligned inputs (N, 10), mask of copied inputs or None).
    Returns per-state arrays: wc (N, 7), wg (N, 8) within-V variances of coordinates and scores (units of M),
    chg / chg_clip (N,) share of the K swaps that change the answer (hybrid as is / clipped at 0),
    neg (N,) share of hybrids with a negative activation, nfix (N,) number of copied inputs.
    part: what the hybrid takes from the resampled state: its "tape" part (default), its "rest", or "all" of it."""
    rng = np.random.default_rng(seed)
    C, G = [P.c], [P.g]
    chg, clip, neg, nfix, acc, rest, whole = np.zeros((7, P.n))
    for _ in range(K):
        xk, mask = sampler(rng)
        hk = run(xk)[:, 9]
        Mk = xk.max(1)                                                    # M > 0 in a quiet state; padding is 0
        C.append(tape_coords(hk) / Mk[:, None]); G.append(tape_scores(hk)[:, :8] / Mk[:, None])
        hk = hk * (P.M / Mk)[:, None]                                     # donor in the recipient's units
        hyb = swap_tape(P.h, hk) if part == "tape" else swap_tape(hk, P.h) if part == "rest" else hk
        nfix += 0 if mask is None else mask.sum(1)
        if not swap:
            continue
        a = finish(hyb, P.x, P.t)
        chg += a != P.ans
        acc += a == P.truth
        clip += finish(np.maximum(hyb, 0), P.x, P.t) != P.ans
        neg += (hyb < -1e-9).any(1)
        if extra:                                                         # the other two transplants, for reference
            rest += finish(swap_tape(hk, P.h), P.x, P.t) != P.ans
            whole += finish(hk, P.x, P.t) != P.ans
    return dict(wc=np.var(C, 0, ddof=1), wg=np.var(G, 0, ddof=1), chg=chg / K, clip=clip / K, neg=neg / K, nfix=nfix / K,
                acc=acc / K, rest=rest / K, whole=whole / K)


def r_seg(g, k):
    """The k largest inputs of one segment: "old" = older than both maximum and runner-up, "mid" = between them,
    "new" = younger than both (k = 10: the whole segment), "pre" = old and mid together = every input that arrived
    before the last record event (the arrival of the younger of maximum and runner-up)."""
    def f(xv):
        two = np.argsort(-xv, 1)[:, :2]
        lo, hi = two.min(1)[:, None], two.max(1)[:, None]
        seg = {"old": COL < lo, "mid": (COL > lo) & (COL < hi), "new": COL > hi, "pre": (COL < hi) & (COL != lo)}[g] & np.isfinite(xv)
        b = -np.sort(-np.where(seg, xv, -np.inf), 1)[:, k - 1:k]
        return seg & (xv >= b), np.where(seg, b, np.inf)
    return f


def V(P, *rules):
    """Sampler for a candidate V given as rules (the maximum and the runner-up are always part of it)."""
    rules = (r_top(2),) + rules
    return lambda rng: resample(P.xr, P.real, rules, rng)


def summary(P, r, q=None):
    """FU per coordinate, FU of the summed variance, FU of the scores (ages >= 2), answer-change rates (%)."""
    q = np.ones(P.n, bool) if q is None else q
    vc, vg = P.c.var(0), P.g.var(0)                                       # denominators: all quiet states
    return dict(fu=r["wc"][q].mean(0) / vc, fu_sum=r["wc"][q].mean(0).sum() / vc.sum(),
                fu_sc=r["wg"][q].mean(0).sum() / vg.sum(), chg=100 * r["chg"][q].mean(), clip=100 * r["clip"][q].mean(),
                neg=100 * r["neg"][q].mean(), nfix=r["nfix"][q].mean())


HEAD = f"{'candidate V':46s} {'kept':>5s} | " + " ".join(f"{c:>8s}" for c in COORD) + " |  FU sum  FU score | answer changed %  (clipped)  neg %"


def line(name, s):
    return (f"{name:46s} {s['nfix']:5.2f} | " + " ".join(f"{v:8.3f}" for v in s["fu"])
            + f" | {s['fu_sum']:7.3f}  {s['fu_sc']:7.3f}  | {s['chg']:10.2f}       {s['clip']:6.2f}   {s['neg']:6.1f}")
