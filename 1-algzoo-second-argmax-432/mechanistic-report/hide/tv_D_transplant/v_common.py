"""Verifier's own helpers for the transplant investigation (independent of tape_common.py and of t1_transplants/tp.py).

The tape projector is built here directly from the eigen-decomposition of A = W[o, o] (modes picked by eigenvalue),
and compared with tape_common.P_TAPE in __main__.

  st = stats(x, t): top three of x_0..x_t (values M, S, T; positions pM, pS, pT; ages aM, aS, aT) and the quiet mask
  hyb(hA, hB)     : A's state with its tape part replaced by B's   (rest of A + tape of B)
  ans_at(h, t)    : position named by the scores at step t (entry i of U h belongs to position i - (9 - t))
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True
import numpy as np
from common import w, W, U, run, ORD, sample

O = np.array(ORD)
A = W[np.ix_(O, O)]
lam, V = np.linalg.eig(A)
L = np.linalg.inv(V)
MOD, ANG = np.abs(lam), np.abs(np.degrees(np.angle(lam)))
TARGET = {"level": (1.0099, 0), "runner": (0.9735, 0), "clock": (0.951, 42.75), "h2": (0.937, 83.28), "alt": (0.928, 180),
          "h3": (0.837, 124.78), "short": (0.735, 42.99), "trans": (0.231, 0), "echo": (0.001, 0)}
IDX = {n: [k for k in range(13) if abs(MOD[k] - m) < 5e-3 and abs(ANG[k] - a) < 0.5] for n, (m, a) in TARGET.items()}
assert sorted(sum(IDX.values(), [])) == list(range(13)), IDX
TAPE = ["alt", "short", "h2", "h3"]


def proj(names):
    """13 x 13 real projector on the given modes (both members of each complex pair)."""
    idx = sum((IDX[n] for n in names), [])
    return np.real(V[:, idx] @ L[idx, :])


P_T = proj(TAPE)


def part(H, P=P_T):
    """(..., 16) -> (..., 16): the part of the state in the projector's modes (zero on the pulse units)."""
    out = np.zeros(H.shape)
    out[..., O] = H[..., O] @ P.T
    return out


def hyb(hA, hB, P=P_T, q=1.0):
    """A's state with the P-part moved a fraction q of the way to B's P-part."""
    return hA + q * (part(hB, P) - part(hA, P))


def state(x, t):
    return run(x[:, : t + 1])[:, -1]


def final(h, fut=None, clip=False):
    """Final answer (position 0..9) from a state, after the further inputs fut (B, K)."""
    if clip:
        h = np.maximum(h, 0)
    if fut is not None and fut.shape[1]:
        h = run(fut, h0=h)[:, -1]
    return (h @ U.T).argmax(1)


def ans_at(h, t, clip=False):
    """Running answer at step t: the position (0..t) with the largest score."""
    if clip:
        h = np.maximum(h, 0)
    return (h @ U.T)[:, 9 - t:].argmax(1)


def stats(x, t):
    pre = x[:, : t + 1]
    order = np.argsort(-pre, 1, kind="stable")
    ar = np.arange(len(x))
    st = {}
    for k, nm in enumerate("MST"):
        p = order[:, k]
        st["p" + nm], st[nm], st["a" + nm] = p, pre[ar, p], t - p
    st["quiet"] = (st["aM"] >= 2) & (st["aS"] >= 2) & (st["S"] > 0)
    return st


def quiet(n, t, seed, dist="normal"):
    x = sample(n, seed, dist)
    x = x[stats(x, t)["quiet"]]
    return x, stats(x, t)


def sub(d, m):
    return {k: v[m] for k, v in d.items()}


def far_low(x, st, t, far=0.4):
    """(B, t+1) bool: inputs more than far*M below the runner-up, the top three excluded."""
    pre, ar = x[:, : t + 1], np.arange(len(x))
    low = pre < (st["S"] - far * st["M"])[:, None]
    for k in ("pM", "pS", "pT"):
        low[ar, st[k]] = False
    return low


def pick_low_slot(x, st, t, rng):
    """A random far-below slot of age >= 2 per row, and whether one exists."""
    low = far_low(x, st, t)
    low[:, max(t - 1, 0):] = False
    return np.where(low, rng.random(low.shape), -1).argmax(1), low.any(1)


def swapped(x, p, j):
    ar, xB = np.arange(len(x)), x.copy()
    xB[ar, p], xB[ar, j] = x[ar, j], x[ar, p]
    return xB


def lows(M, K, rng):
    """K negative inputs per row, -M * U(0.05, 0.6)."""
    return -M[:, None] * rng.uniform(0.05, 0.6, (len(M), K))


def shares(ans, names):
    out = {k: np.mean(ans == p) for k, p in names.items()}
    out["other"] = np.mean(~np.any([ans == p for p in names.values()], 0))
    return "  ".join(f"{k} {v:.3f}" for k, v in out.items())


if __name__ == "__main__":
    import tape_common as tc
    print("modes picked:", {n: [f"{MOD[k]:.4f}@{np.degrees(np.angle(lam[k])):.2f}" for k in IDX[n]] for n in TAPE})
    print("own projector vs tape_common.P_TAPE, max abs difference:", np.abs(P_T - tc.P_TAPE).max())
    print("projector: P P = P", np.abs(P_T @ P_T - P_T).max(), " rank", np.linalg.matrix_rank(P_T), " imaginary part dropped:",
          np.abs(np.imag(V[:, sum((IDX[n] for n in TAPE), [])] @ L[sum((IDX[n] for n in TAPE), []), :])).max())
    x = sample(20000, 5)
    h = state(x, 9)
    print("part() vs tape_common.tape_part:", np.abs(part(h) - tc.tape_part(h)).max())
    print("hyb() vs tape_common.swap_tape:", np.abs(hyb(h, h[::-1]) - tc.swap_tape(h, h[::-1])).max())
    st, st2 = stats(x, 9), tc.prefix_stats(x, 9)
    print("quiet mask equal:", np.array_equal(st["quiet"], st2["quiet"]), " share", st["quiet"].mean())
    xz = np.zeros_like(x); xz[:, 3:] = x[:, :7]
    print("zero-prefix identity (h_6 = h_9 of padded):", np.abs(state(xz, 9) - state(x, 6)).max())
    print("running answer at t = 6 equals final answer of the padded sequence minus 3:",
          np.mean(ans_at(state(x, 6), 6) == final(state(xz, 9)) - 3))
