"""S1. FU(V) and the resample-and-swap test for the candidate sets V0..V5 and relatives.
Population: every quiet state (a_M >= 2, a_S >= 2, S > 0) at steps t = 3..9 of 80,000 N(0,1) sequences.
V always contains the step t, the ages and values of maximum and runner-up (V0).  K = 4 resamples per state.
Run:  python -B s1_candidates.py > s1_candidates.txt"""
from rs import *

P = Pop(80_000, seed=1)
print(f"{P.n} quiet states from 80000 sequences; per step t = 3..9: {np.bincount(P.t)[3:]}")
print(f"network answers the runner-up of the whole sequence in {100 * np.mean(P.ans == P.truth):.2f}% of them")
print("tape coordinates / M over these states: sd", np.round(P.c.std(0), 2), " mean", np.round(P.c.mean(0), 2))
print("\n'kept' = mean number of inputs whose age and value are in V (a state has t + 1 = "
      f"{(P.t + 1).mean():.2f} inputs on average).  FU = unexplained fraction of variance (0 = V determines the tape).")
print("'answer changed' = share of states whose final answer changes when the tape part is replaced by that of a resampled state.\n")


def fired5(xr):
    """(N, 10) bool: the input in column j made unit 5 fire one step later (columns 0..8)."""
    f = np.zeros(xr.shape, bool)
    f[:, :9] = run(xr)[:, 1:, 5] > 0
    return f


F5 = fired5(P.xr) & P.real
V4 = (r_top(2), r_records(2, -np.inf))
FAIL = []


def v5(rng, rounds=200):
    """V5 = V4 + every input that fired unit 5 one step after it arrived.  Rejection: redraw the other inputs below
    their record bound until the set of inputs that fire unit 5 is the original one."""
    mask, bound = fixed(np.where(P.real, P.xr, -np.inf), V4)
    free = P.real & ~mask & ~F5
    out, todo = P.xr.copy(), np.arange(P.n)
    for _ in range(rounds):
        new = np.where(free[todo], ndtri(rng.uniform(1e-300, 1, (len(todo), 10)) * ndtr(bound[todo])), P.xr[todo])
        ok = ((fired5(new) & P.real[todo]) == F5[todo]).all(1)
        out[todo[ok]] = new[ok]
        todo = todo[~ok]
        if len(todo) == 0:
            break
    FAIL.append(len(todo) / P.n)
    return out, mask | F5


perm = np.random.default_rng(5).permutation(P.n)
cands = [
    ("sham: own tape put back", lambda rng: (P.xr, P.real)),
    ("unrelated quiet state (any t, rescaled to M)", lambda rng: (P.xr[rng.permutation(P.n)], None)),
    ("V0 = max and runner-up (ages, S/M)", V(P)),
    ("V1 = V0 + third largest", V(P, r_top(3))),
    ("V2 = V1 + fourth largest", V(P, r_top(4))),
    ("     top 5", V(P, r_top(5))),
    ("     top 6", V(P, r_top(6))),
    ("V3(0.9)  = all inputs > 0.9 S", V(P, r_above(0.9))),
    ("V3(0.75) = all inputs > 0.75 S", V(P, r_above(0.75))),
    ("V3(0.5)  = all inputs > 0.5 S", V(P, r_above(0.5))),
    ("V3(0.25) = all inputs > 0.25 S", V(P, r_above(0.25))),
    ("V3(0)    = all positive inputs", V(P, r_above(0.0))),
    ("V4 = record history (top 2 on arrival)", V(P, r_records(2, -np.inf))),
    ("V4z = same, records above zero only", V(P, r_records(2))),
    ("     earlier maxima only (above zero)", V(P, r_records(1))),
    ("     top 3 on arrival (above zero)", V(P, r_records(3))),
    ("V5 = V4 + inputs that fired unit 5", v5),
    ("     inputs > 0.75 x runner-up on arrival", V(P, r_runthr(0.75))),
    ("     inputs > 0.5 x runner-up on arrival", V(P, r_runthr(0.5))),
    ("V4z + third largest", V(P, r_records(2), r_top(3))),
    ("V4z + third + fourth largest", V(P, r_records(2), r_top(4))),
    ("V4z + all inputs > 0.5 S", V(P, r_records(2), r_above(0.5))),
    ("V4z + all positive inputs", V(P, r_records(2), r_above(0.0))),
    ("V0 + inputs of age 0, 1", V(P, r_ages([0, 1]))),
    ("V0 + inputs of age 0..3", V(P, r_ages([0, 1, 2, 3]))),
]
print(HEAD)
res = {}
for name, s in cands:
    res[name] = evaluate(P, s)
    print(line(name, summary(P, res[name])), flush=True)
print(f"\nV5: share of states where the rejection sampler found no resample in 200 rounds (original kept): {np.mean(FAIL):.4f}")
print(f"V5: inputs that fired unit 5 per state {F5.sum(1).mean():.2f}; of these not already in V4: "
      f"{(F5 & ~fixed(np.where(P.real, P.xr, -np.inf), V4)[0]).sum(1).mean():.2f}")

print("\nReference: what the same swaps do when the REST of the state (or the whole state) is taken from the resampled state instead")
print(HEAD)
for name in ("V0 = max and runner-up (ages, S/M)", "V1 = V0 + third largest", "V3(0)    = all positive inputs"):
    s = dict(cands)[name]
    for part in ("rest", "all"):
        print(line(f"{part:4s} | {name[:38]}", summary(P, evaluate(P, s, part=part))), flush=True)

print("\nFinal states only (t = 9), same candidates")
q = P.t == 9
print(f"{q.sum()} states")
print(HEAD)
for name, _ in cands:
    print(line(name, summary(P, res[name], q)))
