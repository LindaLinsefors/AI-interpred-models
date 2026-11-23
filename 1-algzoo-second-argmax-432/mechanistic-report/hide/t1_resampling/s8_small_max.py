"""S8. Robustness of FU: c / M is heavy-tailed when the maximum M_t is small against the other inputs (which are
N(0,1) whatever M_t is), so a few states can carry much of a variance.  FU and the tape-transplant answer-change rate
by M_t, and the share of E[Var(c | V)] that comes from the 1% of states with the largest within-V variance.
Two populations (40,000 sequences each, seeds 2 and 5), K = 4.
Run:  python -B s8_small_max.py > s8_small_max.txt"""
from rs import *

CANDS = (("V0 = max and runner-up", ()), ("V0 + D", (r_seg("pre", 1),)), ("records above zero (c = 1)", (r_records(2),)),
         ("contenders c = 0.75", (r_runthr(0.75),)), ("all positive inputs (c = 0)", (r_runthr(0.0),)))
BINS = ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 9))
for seed in (2, 5):
    P = Pop(40_000, seed=seed)
    print(f"population seed {seed}: {P.n} quiet states; share with M_t in " + ", ".join(f"({a}, {b}]: {np.mean((P.M > a) & (P.M <= b)):.3f}" for a, b in BINS))
    print(f"  summed variance of c/M: all states {P.c.var(0).sum():.2f}; " + "; ".join(f"M in ({a}, {b}]: {P.c[(P.M > a) & (P.M <= b)].var(0).sum():.2f}" for a, b in BINS))
    print(f"  {'V':30s} | FU all | top-1% share |  FU within M-bin (numerator and denominator from that bin)   | FU, M > 0.5 | answer changed % per M-bin")
    for name, rules in CANDS:
        r = evaluate(P, V(P, *rules))
        w = r["wc"].sum(1)
        top = np.sort(w)[-P.n // 100:].sum() / w.sum()
        q5 = P.M > 0.5
        fb = [w[(P.M > a) & (P.M <= b)].mean() / P.c[(P.M > a) & (P.M <= b)].var(0).sum() for a, b in BINS]
        cb = [100 * r["chg"][(P.M > a) & (P.M <= b)].mean() for a, b in BINS]
        print(f"  {name:30s} | {w.mean() / P.c.var(0).sum():6.4f} |    {top:6.3f}    |  " + "  ".join(f"{v:8.4f}" for v in fb)
              + f"          |   {w[q5].mean() / P.c[q5].var(0).sum():6.4f}    | " + " ".join(f"{v:6.2f}" for v in cb))
    print()
