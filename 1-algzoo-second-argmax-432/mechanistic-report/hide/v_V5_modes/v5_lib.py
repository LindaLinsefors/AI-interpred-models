"""Verifier V5: shared helpers (own code). Data are regenerated on the fly; nothing is cached on disk."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *      # noqa
from modes import *       # noqa

np.set_printoptions(precision=3, suppress=True, linewidth=220)
W4 = w[4]                                   # 10.157
OMEGA = float(np.angle(LAM["clock"]))       # 42.75 deg in rad
CI = [ORD.index(j) for j in COMP]
WI = [ORD.index(j) for j in WAVE]
Uo = U[:, ORD]
SEEDS = [5010, 5011, 5012, 5013]            # 4 x 200k sequences = 800k sequences, 8M states
NB = 200_000


def top3(x):
    """Own incremental running top-3 of the prefix x_0..x_t (inclusive). Returns dict of (B, T) arrays:
    M, S, T3 (values, -inf if absent), pM, pS, pT (positions, -1 if absent), a1, a2, a3 (ages, -1 if absent)."""
    B, Tn = x.shape
    ninf = -np.inf
    M = np.full(B, ninf); S = np.full(B, ninf); T3 = np.full(B, ninf)
    pM = np.full(B, -1); pS = np.full(B, -1); pT = np.full(B, -1)
    out = {k: np.empty((B, Tn)) for k in ("M", "S", "T3")}
    out.update({k: np.empty((B, Tn), dtype=int) for k in ("pM", "pS", "pT")})
    for t in range(Tn):
        v = x[:, t]
        nm = v > M
        ns = ~nm & (v > S)
        nt = ~nm & ~ns & (v > T3)
        T3 = np.where(nm | ns, S, np.where(nt, v, T3)); pT = np.where(nm | ns, pS, np.where(nt, t, pT))
        S = np.where(nm, M, np.where(ns, v, S)); pS = np.where(nm, pM, np.where(ns, t, pS))
        M = np.where(nm, v, M); pM = np.where(nm, t, pM)
        for k, a in (("M", M), ("S", S), ("T3", T3), ("pM", pM), ("pS", pS), ("pT", pT)):
            out[k][:, t] = a
    tt = np.arange(Tn)[None, :]
    for k, p in (("a1", "pM"), ("a2", "pS"), ("a3", "pT")):
        out[k] = np.where(out[p] >= 0, tt - out[p], -1)
    return out


def batch(seed, n=NB):
    """x, H, Z, stats, mode coordinates (dict name -> (B, T))."""
    x = sample(n, seed)
    H, Z = run(x, return_pre=True)
    st = top3(x)
    m = coords(H)
    return x, H, Z, st, m


def lsq(X, y):
    """Least squares without intercept via normal equations; X real (n, k), y real or complex (n,).
    Returns coef, uncentred R2 = 1 - sum|res|^2 / sum|y|^2, centred R2 = 1 - sum|res|^2 / sum|y - mean y|^2."""
    G = X.T @ X
    c = np.linalg.solve(G, X.T @ y)
    res = y - X @ c
    ss = (np.abs(res) ** 2).sum()
    return c, 1 - ss / (np.abs(y) ** 2).sum(), 1 - ss / (np.abs(y - y.mean()) ** 2).sum()


def pol(c):
    return f"{abs(c):.3f}@{np.degrees(np.angle(c)):+7.1f}"


def wrap(a):
    """wrap angle (rad) to (-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    # cross-check own top3 against common.running_stats
    x = sample(20000, 5003)
    a, b = top3(x), running_stats(x)
    for k1, k2 in (("M", "M1"), ("S", "M2"), ("T3", "M3"), ("pM", "p1"), ("pS", "p2"), ("pT", "p3"), ("a1", "a1"), ("a2", "a2"), ("a3", "a3")):
        print(k1, k2, "identical:", bool(np.array_equal(a[k1], b[k2])))
    print("w4 =", W4, " omega (deg) =", np.degrees(OMEGA))
