"""a7: a tape that is a function of a few named quantities, transplanted into real states.

Recipient A: a quiet state of an ordinary N(0,1) sequence.  Donor B: an artificial sequence that keeps only some of
A's inputs (same positions, same values) and has ZERO everywhere else in the prefix:
   M, S only            -> B's tape is a function of (a_M, a_S, S/M) times M
   M, S, value 0.6 M    -> runner-up's value replaced by 0.6 M: a function of (a_M, a_S) times M
   S only               -> only the runner-up kept (B's maximum is then A's runner-up; B is not a quiet state)
   M only               -> only the maximum kept
   top 3 only           -> function of the three positions and S/M, T/M
   top 4 only
   nothing (tape = 0)   -> the tape part deleted
hybrid = A's rest + B's tape.  Accuracy = P(answer = correct position for A's prefix + future) [clipped at zero],
A intact for comparison.  Transplant at t = 9 (read out at once) and at t = 6 with chosen futures.
"""
from tp import *

N = 600_000


def keep_only(x, t, pos_list, values=None):
    xB = x.copy(); xB[:, : t + 1] = 0
    ar = np.arange(len(x))
    for i, p in enumerate(pos_list):
        xB[ar, p] = x[ar, p] if values is None or values[i] is None else values[i]
    return xB


for t in (9, 6):
    x, st = quiet_sample(N, t, seed=71)
    hA = state_at(x, t)
    rng = np.random.default_rng(72 + t)
    n = len(x)
    pre = x[:, : t + 1]
    p4 = np.argsort(-pre, 1, kind="stable")[:, 3]
    low = -st["M"][:, None] * rng.uniform(0.05, 0.6, (n, 9 - t))
    put = lambda v, k: np.concatenate([low[:, :k], v[:, None], low[:, k + 1:]], 1)
    if t == 9:
        futs = {"read out at once": low}
    else:
        futs = {"low x3": low, "new max at 7": put(1.3 * st["M"], 0), "new runner-up at 7": put((st["S"] + st["M"]) / 2, 0),
                "new third at 7": put((np.maximum(st["T3"], 0) + st["S"]) / 2, 0), "A's own inputs": x[:, t + 1:]}
    donors = {"identical (A intact)": x,
              "top 4 only": keep_only(x, t, [st["pM"], st["pS"], st["pT"], p4]),
              "top 3 only": keep_only(x, t, [st["pM"], st["pS"], st["pT"]]),
              "M, S only": keep_only(x, t, [st["pM"], st["pS"]]),
              "M, S, value 0.6 M": keep_only(x, t, [st["pM"], st["pS"]], [None, 0.6 * st["M"]]),
              "S only": keep_only(x, t, [st["pS"]]),
              "M only": keep_only(x, t, [st["pM"]]),
              "nothing (tape = 0)": keep_only(x, t, [])}
    near = st["T3"] > st["S"] - NEAR * st["M"]
    print(f"\ntransplant at t = {t}, {n} quiet states;  third within 0.1 M of the runner-up in {near.mean():.3f} of them")
    print("   " + " " * 22 + "".join(f"{k:>24s}" for k in futs) + "   | t-readout split: third near / not near | negative entry")
    for lab, xB in donors.items():
        hyb = hybrid(hA, state_at(xB, t))
        row = f"   {lab:22s}"
        for f in futs.values():
            tA = second_argmax(np.concatenate([pre, f], 1))
            row += f"   {np.mean(finish(hyb, f) == tA):.3f} [{np.mean(finish(hyb, f, True) == tA):.3f}]      "
        f0 = next(iter(futs.values()))
        ok0 = finish(hyb, f0) == second_argmax(np.concatenate([pre, f0], 1))
        print(row + f"   | {ok0[near].mean():.3f} / {ok0[~near].mean():.3f} | {neg_share(hyb):.3f}")
