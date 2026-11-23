"""S1. Which other parts complete the tape piece?  All 64 subsets of {level, runner, clock, trans, echo, pulse}.
For each subset: scores = tape piece + the pieces of the subset.  Measures on quiet states (see t1_common):
  P(S top)  = the runner-up's entry is the largest among existing entries of age >= 2
  agree     = the largest such entry is the same as the full score's
  slope, err_sd, err_iqr = comparison near the runner-up (entries within 0.1 M below S): slope of
              (entry - runner-up's entry)/M against (x - S)/M, and the residual spread divided by the slope
              (an error in the input, units of M; sd and half the interquartile range)
Run:  python -B s1_subsets.py > s1_subsets.txt"""
from itertools import combinations

from t1_common import *

for label, steps, B, seed in (("QUIET FINAL STATES (t = 9)", (9,), 600_000, 101), ("QUIET STATES, ALL STEPS t = 3..9", range(3, 10), 150_000, 102)):
    d = population(B, seed, steps); n = len(d["h"])
    pc = pieces(d["h"]); full = d["h"] @ U.T
    assert np.abs(sum(pc.values()) - full).max() < 1e-8
    top_full = top_entry(full, d["valid"])
    tape = sum(pc[m] for m in TAPE)
    print(f"\n================ {label}: {n} states from {B} sequences (seed {seed})")
    print("who is on top among ages >= 2:           runner-up   maximum   third   other")
    for nm, A in (("tape piece", tape), ("full score", full)):
        top = top_entry(A, d["valid"])
        print(f"  {nm:38s} {np.mean(top == d['iS']):.3f}      {np.mean(top == d['iM']):.3f}    {np.mean(top == d['iT']):.3f}   "
              f"{np.mean((top != d['iS']) & (top != d['iM']) & (top != d['iT'])):.3f}")
    res = {}
    for k in range(7):
        for sub in combinations(OTHER, k):
            res[sub] = measures(tape + sum(pc[m] for m in sub), d, top_full)
    print("\nsubset added to the tape piece                 P(S top)  agree   slope  err_sd  err_iqr")
    for sub, r in sorted(res.items(), key=lambda kv: (len(kv[0]), -kv[1][0])):
        print(f"  {len(sub)}  {'+'.join(sub) if sub else '(tape piece alone)':42s} {r[0]:.3f}    {r[1]:.3f}  {r[2]:6.1f}  {r[3]:.4f}  {r[4]:.4f}")
    print("\nbest subset of each size by P(S top):")
    for k in range(7):
        sub, r = max(((s, r) for s, r in res.items() if len(s) == k), key=lambda kv: kv[1][0])
        print(f"  size {k}: {'+'.join(sub) if sub else '(none)':40s} P(S top) {r[0]:.3f}  agree {r[1]:.3f}  err_iqr {r[4]:.4f}")
    print("\nthe other direction: each part ALONE and the five-mode rest alone (no tape piece)")
    for nm, A in [(m, pc[m]) for m in PARTS] + [("rest = level+runner+clock+trans+echo+pulse", full - tape), ("level+runner+clock", pc["level"] + pc["runner"] + pc["clock"])]:
        r = measures(A, d, top_full)
        top = top_entry(A, d["valid"])
        print(f"  {nm:44s} P(S top) {r[0]:.3f}  P(M top) {np.mean(top == d['iM']):.3f}  agree {r[1]:.3f}  slope {r[2]:6.1f}")
    print("\nleave one part out of the FULL score:")
    for m in PARTS:
        r = measures(full - pc[m], d, top_full)
        print(f"  without {m:8s} P(S top) {r[0]:.3f}  agree {r[1]:.3f}  slope {r[2]:6.1f}  err_sd {r[3]:.4f}  err_iqr {r[4]:.4f}")
    if len(steps) > 1:
        continue
    # ---- by configuration: is the completing part the same everywhere?
    singles = {m: top_entry(tape + pc[m], d["valid"]) == d["iS"] for m in OTHER}
    t0 = top_entry(tape, d["valid"]) == d["iS"]; tf = top_full == d["iS"]
    t3 = top_entry(tape + pc["level"] + pc["runner"] + pc["clock"], d["valid"]) == d["iS"]
    print("\nP(S top) by order of arrival and by age of the maximum (rows) -- columns: n, tape alone, + each single part, +level+runner+clock, full")
    hdr = "                      n     tape   " + "  ".join(f"{m:>6s}" for m in OTHER) + "    +LRC   full   best single"
    for order_name, om in (("runner-up arrived BEFORE the maximum (aS > aM)", d["aS"] > d["aM"]), ("runner-up arrived AFTER the maximum (aS < aM)", d["aS"] < d["aM"])):
        print(f" {order_name}: {om.mean():.3f} of states\n{hdr}")
        for key, arr in (("aM", d["aM"]), ("aS", d["aS"])):
            for a in range(2, 10):
                q = om & (arr == a)
                if q.sum() < 500:
                    continue
                vals = [singles[m][q].mean() for m in OTHER]
                print(f"   {key} = {a}         {q.sum():7d}   {t0[q].mean():.3f}  " + "  ".join(f"{v:6.3f}" for v in vals)
                      + f"   {t3[q].mean():.3f}  {tf[q].mean():.3f}   {OTHER[int(np.argmax(vals))]}")
    print("\nbest single part in each of the 56 cells (aM, aS): count of cells won by each part, and the gain range")
    win = {m: 0 for m in OTHER}; gains = []
    print("   rows aM = 2..9, columns aS = 2..9: best single part (first letter; L level, R runner, C clock, T trans, E echo, P pulse) and tape -> best -> full")
    for aM in range(2, 10):
        row = f"   aM={aM}: "
        for aS in range(2, 10):
            q = (d["aM"] == aM) & (d["aS"] == aS)
            if aM == aS or q.sum() < 300:
                row += "       --        "; continue
            vals = [singles[m][q].mean() for m in OTHER]; b = int(np.argmax(vals)); win[OTHER[b]] += 1
            gains.append((t0[q].mean(), vals[b], tf[q].mean()))
            row += f" {OTHER[b][0].upper()} {t0[q].mean():.2f}>{vals[b]:.2f}>{tf[q].mean():.2f}"
        print(row)
    print("   cells won:", win)
