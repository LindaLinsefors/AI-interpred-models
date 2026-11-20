"""Verifier V2 (two score vectors): shared setup. Own code, written from the claims file only.

    s_t = U h_t,  V_t = U[:, COMP] h_t[COMP],  m_t = s_t - V_t.
    Index i of a score vector at step t refers to position p = i + t - 9 (age a = 9 - i).
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *      # noqa: E402,F401
from modes import *       # noqa: E402,F401

OMEGA = float(np.angle(LAM["clock"]))            # 42.75 deg in radians
NONCOMP = [u for u in range(N) if u not in COMP]
UC = U[:, COMP]                                  # (10, 4)
B_DEFAULT = 200_000


def top2(x):
    """Own order statistics of every prefix x_0..x_t: M, S (B, T), p1, p2 (B, T). S = -inf, p2 = -1 at t = 0."""
    Bn, Tn = x.shape
    M = np.empty((Bn, Tn)); S = np.full((Bn, Tn), -np.inf)
    p1 = np.empty((Bn, Tn), dtype=int); p2 = np.full((Bn, Tn), -1, dtype=int)
    rows = np.arange(Bn)
    for t in range(Tn):
        pre = x[:, : t + 1].copy()
        p1[:, t] = pre.argmax(1)
        M[:, t] = pre[rows, p1[:, t]]
        if t > 0:
            pre[rows, p1[:, t]] = -np.inf
            p2[:, t] = pre.argmax(1)
            S[:, t] = pre[rows, p2[:, t]]
    return M, S, p1, p2


def batches(n_batches, seed0, B=B_DEFAULT):
    """Yield (x, H, M, S, p1, p2) for seeds seed0 .. seed0 + n_batches - 1."""
    for k in range(n_batches):
        x = sample(B, seed=seed0 + k)
        yield (x, run(x)) + top2(x)


def scores(h):
    """h (B, 16) -> s, V, m (B, 10)."""
    s = h @ U.T
    V = h[:, COMP] @ UC.T
    return s, V, s - V


if __name__ == "__main__":
    print("omega (deg) =", np.degrees(OMEGA), " |lam| =", abs(LAM["clock"]))
    x = sample(50_000, seed=4999)
    M, S, p1, p2 = top2(x)
    st = running_stats(x)
    ok = (np.array_equal(p1, st["p1"]) and np.array_equal(p2, st["p2"]) and np.array_equal(M, st["M1"])
          and np.array_equal(S, st["M2"]))
    print("own top2 == common.running_stats:", ok)
    print("accuracy on this batch:", accuracy(x))
    print("LEFT[clock] @ W[ORD, 4] =", LEFT["clock"] @ W[ORD, 4])
