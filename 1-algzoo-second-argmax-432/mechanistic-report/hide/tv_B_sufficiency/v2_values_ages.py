"""V2. Values against positions (claim 1) and the two halves of D (claim 3), with samplers written from scratch.
Every line fixes the step t and "the state is quiet" and redraws the rest from the exact conditional distribution:
  nothing            : whole rows redrawn iid N(0,1) until the state is quiet (brute-force rejection)
  ages of M, S only  : whole rows redrawn until maximum and runner-up sit in the original columns and S > 0 (rejection)
  S/M only           : M and S keep their values and get new columns of age >= 2; the others iid below S
  all ages by rank   : new values (rows redrawn until S > 0), placed so that the rank of every column is unchanged
  all values, no ages: the real entries are permuted at random until maximum and runner-up have age >= 2 (rejection)
  V0 + age of D      : the other inputs iid below S, then the largest early one is swapped into D's column
  V0 + value of D    : early inputs iid below D, later ones below S, D put at a random early column
D = the largest input that arrived before the younger of maximum and runner-up (not counting the older of the two).
Population: quiet states at steps 3..9 of 40,000 sequences (seed 22), K = 3.
Run:  python -B v2_values_ages.py > v2_values_ages.txt"""
from vb import *

P = Pop(40_000, seed=22)
N, ar = P.n, np.arange(P.n)
two = np.sort(np.stack([P.cM, P.cS], 1), 1)
lo, hi = two[:, :1], two[:, 1:]
pre, new = P.real & (COL < hi) & (COL != lo), COL > hi
has = pre.any(1)
dcol = np.where(pre, P.xv, -np.inf).argmax(1)
D = np.where(has, P.xv[ar, dcol], np.nan)


def reject(draw, ok):
    """Redraw rows until ok(rows, index) holds for every state."""
    out, todo = np.empty((N, 10)), np.arange(N)
    for _ in range(5000):
        z = draw(todo)
        good = ok(z, todo)
        out[todo[good]] = z[good]
        todo = todo[~good]
        if len(todo) == 0:
            return out
    raise RuntimeError(f"{len(todo)} states left")


def top2(z, idx):
    srt = np.argsort(np.where(P.real[idx], z, -np.inf), 1)
    return srt[:, -1], srt[:, -2]


def s_nothing(rng):
    def ok(z, idx):
        c1, c2 = top2(z, idx)
        return (c1 < 8) & (c2 < 8) & (z[np.arange(len(idx)), c2] > 0)
    return np.where(P.real, reject(lambda idx: rng.standard_normal((len(idx), 10)), ok), 0.0), None


def s_ages_only(rng):
    def ok(z, idx):
        c1, c2 = top2(z, idx)
        return (c1 == P.cM[idx]) & (c2 == P.cS[idx]) & (z[np.arange(len(idx)), c2] > 0)
    return np.where(P.real, reject(lambda idx: rng.standard_normal((len(idx), 10)), ok), 0.0), None


def s_values_only(rng):
    key = np.where(P.real & (COL < 8), rng.uniform(size=(N, 10)), 9.0)       # two random columns of age >= 2
    c = np.argsort(key, 1)[:, :2]
    out = np.where(P.real, draw_below(np.broadcast_to(P.S[:, None], (N, 10)), P.real, rng), 0.0)
    out[ar, c[:, 0]], out[ar, c[:, 1]] = P.M, P.S
    return out, None


def s_ranks_only(rng):
    def ok(z, idx):
        return np.sort(np.where(P.real[idx], z, -np.inf), 1)[:, -2] > 0
    z = np.where(P.real, reject(lambda idx: rng.standard_normal((len(idx), 10)), ok), -np.inf)
    rank = np.argsort(np.argsort(-P.xv, 1, kind="stable"), 1)                # 0 = maximum; padding ranks last
    return np.where(P.real, np.take_along_axis(-np.sort(-z, 1), rank, 1), 0.0), None


def s_all_values(rng):
    def draw(idx):
        key = np.where(P.real[idx], rng.uniform(size=(len(idx), 10)), -1.0)  # padding stays in front
        src = np.argsort(key, 1)                                             # real columns in random order, padding first
        dst = np.argsort(~P.real[idx], 1, kind="stable")                     # real columns in order first ... so re-map:
        z = np.zeros((len(idx), 10))
        k = P.t[idx] + 1
        for n in np.unique(k):                                               # permute the n real values of each row
            q = k == n
            vals = P.xr[idx[q], 10 - n:]
            perm = np.argsort(rng.uniform(size=vals.shape), 1)
            z[q, 10 - n:] = np.take_along_axis(vals, perm, 1)
        return z
    def ok(z, idx):
        c1, c2 = top2(z, idx)
        return (c1 < 8) & (c2 < 8)
    return reject(draw, ok), None


def s_d_age(rng):
    out = np.where(pre | new, draw_below(np.broadcast_to(P.S[:, None], (N, 10)), pre | new, rng), P.xr)
    m = np.where(pre, out, -np.inf).argmax(1)
    h = np.flatnonzero(has)
    a, b = out[h, m[h]].copy(), out[h, dcol[h]].copy()
    out[h, m[h]], out[h, dcol[h]] = b, a
    return out, None


def s_d_value(rng):
    bound = np.where(pre, np.where(has, D, 0.0)[:, None], P.S[:, None])
    out = np.where(pre | new, draw_below(bound, pre | new, rng), P.xr)
    j = np.where(pre, rng.uniform(size=(N, 10)), 9.0).argmin(1)
    h = np.flatnonzero(has)
    out[h, j[h]] = D[h]
    return out, None


rows = [("nothing (another quiet state at the same t)", s_nothing), ("ages of max and runner-up only", s_ages_only),
        ("values S/M only (ages redrawn)", s_values_only), ("V0 = ages and values of max, runner-up", V(P)),
        ("ages of ALL inputs by rank, no values", s_ranks_only), ("ALL values, no ages", s_all_values),
        ("V0 + age of D (not its value)", s_d_age), ("V0 + value of D (not its age)", s_d_value),
        ("V0 + D (age and value)", V(P, segment("pre", 1))), ("top 3 (V0 + third largest)", V(P, top(3)))]
print(f"{N} quiet states from 40000 sequences (seed 22); D exists in {100 * has.mean():.1f}%\n")

# checks of the samplers that could fail
rng = np.random.default_rng(0)
x1 = s_ages_only(rng)[0]; s1 = np.argsort(np.where(P.real, x1, -np.inf), 1)
print("check 'ages only': max and runner-up in the original columns:", (s1[:, -1] == P.cM).all() and (s1[:, -2] == P.cS).all(),
      "; S' > 0:", (x1[ar, s1[:, -2]] > 0).all(), "; mean S'/M' =", round(float(np.mean(x1[ar, s1[:, -2]] / x1[ar, s1[:, -1]])), 3), "(original", round(float(np.mean(P.S / P.M)), 3), ")")
x1 = s_values_only(rng)[0]; s1 = np.argsort(np.where(P.real, x1, -np.inf), 1)
print("check 'S/M only': M, S unchanged:", np.allclose(x1[ar, s1[:, -1]], P.M) and np.allclose(x1[ar, s1[:, -2]], P.S), "; ages >= 2:", (s1[:, -2:] < 8).all(),
      "; share with a_M > a_S:", round(float(np.mean(s1[:, -1] < s1[:, -2])), 3), "(original", round(float(np.mean(P.aM > P.aS)), 3), ")")
x1 = s_all_values(rng)[0]
print("check 'all values': same multiset of values:", np.allclose(np.sort(x1, 1), np.sort(P.xr, 1)), "; padding untouched:", (x1[~P.real] == 0).all())
x1 = s_ranks_only(rng)[0]
print("check 'ranks only': same rank of every column:", (np.argsort(np.where(P.real, x1, -np.inf), 1) == np.argsort(P.xv, 1)).all())
x1 = s_d_age(rng)[0]
print("check 'age of D': D' in D's column:", (np.where(pre, x1, -np.inf).argmax(1)[has] == dcol[has]).all(), "; V0 unchanged:", np.allclose(x1[ar, P.cM], P.M) and (np.where(pre | new, x1, -9) < P.S[:, None]).all())
x1 = s_d_value(rng)[0]
print("check 'value of D': largest early input = D:", np.allclose(np.where(pre, x1, -np.inf).max(1)[has], D[has]), "\n")

print(HEAD)
for name, s in rows:
    print(line(P, name, evaluate(P, s)), flush=True)
