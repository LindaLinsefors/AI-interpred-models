"""Causal tape transplants: shared constructions (angle "what does the rest of the network read from the tape?").

A = recipient sequence, B = donor sequence, both QUIET at step t and identical except for ONE named thing.
Hybrid = A's state at step t with its tape part replaced by B's (tape_common.set_tape / swap_tape).

Pair kinds (all return xB, ok):   FAR = 0.4, NEAR = 0.1 (in units of M_t)
  "low slot" j = a position of age >= 2, not one of the top three, whose input is far below the runner-up
                 (x_j < S - FAR M).
  S_pos    runner-up swapped with a low slot                    (position of the runner-up)
  S_val    runner-up's value redrawn uniformly in (max(T,0), M) (value of the runner-up)
  M_pos    maximum swapped with a low slot                      (position of the maximum)
  M_val    M' = S + g (M - S), g log-uniform in [1/3, 3]        (value of the maximum)
  T_pos    third within NEAR M of the runner-up (age >= 2), swapped with a low slot
  T_up_lo  third far below (age >= 2) raised to S - 0.05 M      (still third)
  T_up_hi  third far below (age >= 2) raised to S + 0.05 M      (the two swap roles: B's runner-up is A's third)
  low_val  every far-below input (not top three) redrawn from N(0,1) truncated at S - FAR M
  low_pos  the far-below inputs permuted among their positions
  unrel    B = another quiet sequence, rescaled to A's M_t
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True
from scipy.stats import norm
from tape_common import *

FAR, NEAR = 0.4, 0.1
MODE_SEL = {"alt": [0], "short": [1, 2], "h2": [3, 4], "h3": [5, 6]}
KINDS = ["S_pos", "S_val", "M_pos", "M_val", "T_pos", "T_up_lo", "T_up_hi", "low_val", "low_pos", "unrel"]


def quiet_sample(n, t, seed):
    """n iid N(0,1) sequences, reduced to those quiet at step t.  Returns x (all 10 inputs) and prefix stats."""
    x = sample(n, seed)
    x = x[prefix_stats(x, t)["quiet"]]
    return x, prefix_stats(x, t)


def far_mask(x, st, t):
    """(B, t+1) bool: inputs of the prefix that are far below the runner-up and not among the top three."""
    pre, ar = x[:, : t + 1], np.arange(len(x))
    low = pre < (st["S"] - FAR * st["M"])[:, None]
    for k in ("pM", "pS", "pT"):
        low[ar, st[k]] = False
    return low


def low_slot(x, st, t, rng):
    low = far_mask(x, st, t)
    low[:, max(t - 1, 0):] = False                      # age >= 2 only
    return (rng.random(low.shape) * low).argmax(1), low.any(1)


def swap(x, p, j):
    ar, xB = np.arange(len(x)), x.copy()
    xB[ar, p], xB[ar, j] = x[ar, j], x[ar, p]
    return xB


def make_pair(kind, x, st, t, rng):
    """Donor sequences xB (same shape as x; only positions <= t differ, except 'unrel') and the mask of valid rows."""
    ar, M, S, T3 = np.arange(len(x)), st["M"], st["S"], st["T3"]
    xB = x.copy()
    if kind in ("S_pos", "M_pos", "T_pos"):
        j, ok = low_slot(x, st, t, rng)
        if kind == "T_pos":
            ok &= (T3 > S - NEAR * M) & (st["aT"] >= 2)
        xB = swap(x, st["p" + kind[0]], j)
    elif kind == "S_val":
        lo, hi = np.maximum(T3, 0) + 0.02 * M, 0.98 * M
        ok = hi > lo
        xB[ar, st["pS"]] = lo + rng.random(len(x)) * (hi - lo)
    elif kind == "M_val":
        g = np.exp(rng.uniform(-np.log(3), np.log(3), len(x)))
        ok = np.ones(len(x), bool)
        xB[ar, st["pM"]] = S + g * (M - S)
    elif kind in ("T_up_lo", "T_up_hi"):
        d = -0.05 if kind == "T_up_lo" else 0.05
        ok = (T3 < S - FAR * M) & (st["aT"] >= 2) & (M - S > 0.1 * M)
        xB[ar, st["pT"]] = S + d * M
    elif kind == "low_val":
        low = far_mask(x, st, t)
        ok = low.any(1)
        new = norm.ppf(rng.random(low.shape) * norm.cdf(S - FAR * M)[:, None])
        xB[:, : t + 1] = np.where(low, new, x[:, : t + 1])
    elif kind == "low_pos":
        low = far_mask(x, st, t)
        ok = low.sum(1) >= 2
        pos = np.broadcast_to(np.arange(t + 1), low.shape)
        sA = np.argsort(np.where(low, pos, 99), 1)                      # the low slots, in order
        sB = np.argsort(np.where(low, rng.random(low.shape), 99), 1)    # the same slots, shuffled
        k = np.arange(t + 1)[None] < low.sum(1)[:, None]
        pre = x[:, : t + 1]
        vals = np.where(k, np.take_along_axis(pre, sB, 1), np.take_along_axis(pre, sA, 1))
        new = pre.copy(); np.put_along_axis(new, sA, vals, 1)
        xB[:, : t + 1] = new
        ok &= (new != pre).any(1)
    elif kind == "unrel":
        perm = rng.permutation(len(x))
        xB = x[perm] * (M / M[perm])[:, None]
        ok = np.ones(len(x), bool)
    else:
        raise ValueError(kind)
    return xB, ok


def hybrid(hA, hB, q=1.0, modes=None):
    """A's state with tape coordinates (1-q) A + q B on the chosen modes (default: all four)."""
    cA, cB = tape_coords(hA), tape_coords(hB)
    sel = list(range(7)) if modes is None else sum((MODE_SEL[m] for m in modes), [])
    c = cA.copy()
    c[:, sel] += q * (cB[:, sel] - cA[:, sel])
    return set_tape(hA, c)


def finish(h, fut=None, clip=False):
    """Final answer from a state at step t: clip if asked, run the remaining inputs, read out."""
    if clip:
        h = np.maximum(h, 0)
    return answer(h if fut is None or fut.shape[1] == 0 else continue_from(h, fut)[:, -1])


def running_answer(h, t):
    """Position named by the scores at step t (zero-prefix identity): entry i belongs to position i - (9 - t)."""
    s = h @ U.T
    return s[:, 9 - t:].argmax(1)


def neg_share(h):
    return np.mean((h < -1e-9).any(1))


def where_ans(ans, names):
    """names: dict label -> position array.  Returns P(ans = each) and P(ans = none of them)."""
    out = {k: np.mean(ans == p) for k, p in names.items()}
    out["other"] = np.mean(~np.any([ans == p for p in names.values()], 0))
    return out


def fmt(d):
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items())
