"""S2. Roles of values and of positions (ages).  Same population and metrics as S1 (40,000 sequences, K = 4).
Each line fixes some quantities and redraws everything else from its exact conditional distribution given them
(the step t and "the state is quiet" are fixed in every line):
  values are redrawn as order statistics of N(0,1) draws (given S > 0, or truncated below the bound that the kept
  values imply); ages are redrawn by shuffling which input sits at which step (maximum and runner-up keep age >= 2).
Run:  python -B s2_values_vs_ages.py > s2_values_vs_ages.txt"""
from rs import *

P = Pop(40_000, seed=2)
N = P.n
ar = np.arange(N)[:, None]
xv = np.where(P.real, P.xr, -np.inf)
order = np.argsort(-xv, 1, kind="stable")
RANK = np.argsort(order, 1)                                  # rank of each column's input (0 = maximum)
SORTED = np.take_along_axis(xv, order, 1)                    # values, largest first (-inf padding last)
nreal = P.t + 1


def ranks(rng, k, new_ms=False):
    """A random assignment of ranks to the real columns that keeps the columns of the k largest inputs
    (new_ms: instead draw new columns of age >= 2 for maximum and runner-up, then shuffle all the others)."""
    key = rng.uniform(size=(N, 10))
    if new_ms:
        k1 = np.where(P.real & (COL < 8), rng.uniform(size=(N, 10)), 9.0)
        two = np.argsort(k1, 1)[:, :2]
        key[ar, two] = [-2.0, -1.0]
    else:
        key[RANK < k] = (RANK - k)[RANK < k]
    key[~P.real] = 9.0
    return np.argsort(np.argsort(key, 1), 1)


def values(rng, k):
    """Values, largest first: the k largest kept, the others redrawn iid below the k-th (k = 0: all redrawn, S > 0)."""
    if k == 0:
        z = np.where(P.real, rng.standard_normal((N, 10)), -np.inf)
        while True:
            bad = -np.sort(-z, 1)[:, 1] <= 0
            if not bad.any():
                break
            z[bad] = np.where(P.real[bad], rng.standard_normal((bad.sum(), 10)), -np.inf)
        return -np.sort(-z, 1)
    z = ndtri(rng.uniform(1e-300, 1, (N, 10)) * ndtr(SORTED[:, k - 1:k]))
    z[:, :k] = SORTED[:, :k]
    z[COL[None, :] >= nreal[:, None]] = -np.inf
    return -np.sort(-z, 1)


def make(kv, kr, new_ms=False, all_values=False, all_ranks=False):
    """kv: number of largest values kept (10 = all); kr: number of largest inputs whose column (age) is kept."""
    def f(rng):
        vals = SORTED if all_values else values(rng, kv)
        rk = RANK if all_ranks else ranks(rng, kr, new_ms)
        new = np.take_along_axis(vals, rk, 1)
        new[~P.real] = 0.0
        return new, None
    return f


rows = [
    ("nothing (another quiet state at the same t)", make(0, 0, new_ms=True)),
    ("ages of max and runner-up only", make(0, 2)),
    ("values S/M only (ages redrawn)", make(2, 0, new_ms=True)),
    ("V0 = ages and values of max, runner-up", make(2, 2)),
    ("ages of ALL inputs by rank, no values", make(0, 0, all_ranks=True)),
    ("ALL values, no ages", make(0, 0, new_ms=True, all_values=True)),
    ("V0 + rank order of the others (no values)", make(2, 0, all_ranks=True)),
    ("V0 + values of the others (ages shuffled)", make(2, 2, all_values=True)),
    ("V0 + age of the third (not its value)", make(2, 3)),
    ("V0 + value of the third (not its age)", make(3, 2)),
    ("V1 = V0 + age and value of the third", make(3, 3)),
    ("V1 + age of the fourth (not its value)", make(3, 4)),
    ("V1 + value of the fourth (not its age)", make(4, 3)),
    ("V2 = V1 + age and value of the fourth", make(4, 4)),
    ("V1 + rank order of the others (no values)", make(3, 0, all_ranks=True)),
    ("V1 + values of the others (ages shuffled)", make(3, 3, all_values=True)),
]
print(f"{N} quiet states from 40000 sequences (steps 3..9).  Columns as in S1; 'kept' is not defined here.\n")
print(HEAD)
for name, s in rows:
    print(line(name, summary(P, evaluate(P, s))), flush=True)

chk = make(2, 2)(np.random.default_rng(0))[0]
st = stats(chk, P.real)
print("\ncheck of the V0 line: max |M' - M| =", np.abs(st["M"] - P.M).max(), " max |S' - S| =", np.abs(st["S"] - P.S).max(),
      " ages equal:", (st["aM"] == P.aM).all() and (st["aS"] == P.aS).all())
chk = make(0, 0, new_ms=True)(np.random.default_rng(0))[0]
st = stats(chk, P.real)
print("check of the 'nothing' line: all quiet:", ((st["aM"] >= 2) & (st["aS"] >= 2) & (st["S"] > 0)).all(),
      " share with a_M > a_S:", np.mean(st["aM"] > st["aS"]).round(3), "(original:", np.mean(P.aM > P.aS).round(3), ")")
