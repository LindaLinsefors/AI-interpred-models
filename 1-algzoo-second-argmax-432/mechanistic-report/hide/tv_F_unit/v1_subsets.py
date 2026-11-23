"""V1. Claims 1, 2, 4: which parts complete the tape piece; where the value comparison sits; by configuration.
Three populations: N(0,1) at t = 9, N(0,1) at all steps t = 3..9, Unif(-1,1) at t = 9.  Own seeds.
Run:  python -B v1_subsets.py > v1_subsets.txt"""
from itertools import combinations

from vcommon import *

for label, x, steps in (("N(0,1), quiet FINAL states", sample(400_000, 9101), (9,)),
                        ("N(0,1), quiet states at ALL steps t = 3..9", sample(120_000, 9102), range(3, 10)),
                        ("Unif(-1,1), quiet FINAL states", sample(400_000, 9103, "uniform"), (9,))):
    d = population(x, steps); n = len(d["h"]); pc = pieces(d["h"]); full = d["h"] @ U.T
    tape = sum(pc[m] for m in TAPE); rest = full - tape; topf = top_old(full, d["valid"])
    pS = lambda Sc: np.mean(top_old(Sc, d["valid"]) == d["iS"])
    ag = lambda Sc: np.mean(top_old(Sc, d["valid"]) == topf)
    print(f"\n================ {label}: {n} states from {len(x)} sequences")
    for nm, Sc in (("tape piece", tape), ("full score", full)):
        tp = top_old(Sc, d["valid"])
        print(f"  on top among old entries, {nm}: runner-up {np.mean(tp == d['iS']):.3f}  maximum {np.mean(tp == d['iM']):.3f}  third {np.mean(tp == d['iT']):.3f}"
              f"  other {np.mean((tp != d['iS']) & (tp != d['iM']) & (tp != d['iT'])):.3f}")
    res = {sub: (pS(tape + sum(pc[m] for m in sub)), ag(tape + sum(pc[m] for m in sub))) for k in range(1, 7) for sub in combinations(OTHER, k)}
    res[()] = (pS(tape), ag(tape))
    print("  tape + one part:  " + "  ".join(f"{m} {res[(m,)][0]:.3f}" for m in OTHER))
    for k in range(7):
        best = sorted(((v, s) for s, v in res.items() if len(s) == k), reverse=True)[:2]
        print(f"  best of size {k}: " + ";  ".join(f"{'+'.join(s) or '(tape alone)'} P(S top) {v[0]:.3f} agree {v[1]:.3f}" for v, s in best))
    print("  slope near the runner-up (logits per unit of (x - S)/M), input error sd, half-IQR:")
    for nm, Sc in [("tape", tape), ("full", full), ("rest", rest), ("tape+trans", tape + pc["trans"]), ("tape+runner+trans", tape + pc["runner"] + pc["trans"]),
                   ("tape+runner+clock+trans", tape + pc["runner"] + pc["clock"] + pc["trans"])] + [(m, pc[m]) for m in TAPE + OTHER]:
        b, sd, iq, ne = near_slope(Sc, d)
        print(f"     {nm:26s} slope {b:6.1f}   sd {sd:8.4f}   half-IQR {iq:8.4f}   ({ne} entries)")
    if len(steps) > 1 or "Unif" in label:
        continue
    # ---- split of the slope by age of the entry and by order of arrival (attack: does the pooled 2.6 hide a split?)
    i8 = np.broadcast_to(np.arange(8), (n, 8))
    print("  slope by age of the entry (tape / rest / full):")
    for a in range(2, 10):
        e = i8 == 9 - a
        print(f"     age {a}: " + " / ".join(f"{near_slope(Sc, d, extra=e)[0]:6.1f}" for Sc in (tape, rest, full)) + f"   n {near_slope(tape, d, extra=e)[3]}")
    for nm, om in (("runner-up BEFORE the maximum", d["aS"] > d["aM"]), ("runner-up AFTER the maximum", d["aS"] < d["aM"])):
        print(f"     {nm}: " + " / ".join(f"{near_slope(Sc, d, extra=om[:, None] & (i8 >= 0))[0]:6.1f}" for Sc in (tape, rest, full)))
    print("  slope of the rest piece in the 56 cells (a_M, a_S) with >= 300 near entries: ", end="")
    sl = []
    for aM in range(2, 10):
        for aS in range(2, 10):
            e = ((d["aM"] == aM) & (d["aS"] == aS))[:, None] & (i8 >= 0)
            if aM != aS and (e & d["valid"] & ((d["xr"][:, :8] - d["S"][:, None]) / d["M"][:, None] < 0) & ((d["xr"][:, :8] - d["S"][:, None]) / d["M"][:, None] > -0.1)).sum() >= 300:
                sl.append((near_slope(rest, d, extra=e)[0], near_slope(tape, d, extra=e)[0]))
    sl = np.array(sl)
    print(f"{len(sl)} cells; rest slope min {sl[:, 0].min():.1f} median {np.median(sl[:, 0]):.1f} max {sl[:, 0].max():.1f};  tape slope min {sl[:, 1].min():.1f} median {np.median(sl[:, 1]):.1f} max {sl[:, 1].max():.1f}")
    # ---- by configuration
    single = {m: top_old(tape + pc[m], d["valid"]) == d["iS"] for m in OTHER}
    t0 = top_old(tape, d["valid"]) == d["iS"]; tf = topf == d["iS"]
    after = d["aS"] < d["aM"]
    print("  by configuration:  n, tape, +trans, +runner, +clock, full")
    for nm, q in [("after, a_S = 2", after & (d["aS"] == 2)), ("after, a_S = 3", after & (d["aS"] == 3)), ("after, a_S = 4", after & (d["aS"] == 4))] + \
                 [(f"before, a_M = {a}", ~after & (d["aM"] == a)) for a in (3, 4, 5, 6)] + [(f"(a_M, a_S) = {c}", (d["aM"] == c[0]) & (d["aS"] == c[1])) for c in ((9, 2), (8, 2), (4, 2))]:
        print(f"     {nm:22s} {q.sum():7d}  {t0[q].mean():.3f}  {single['trans'][q].mean():.3f}  {single['runner'][q].mean():.3f}  {single['clock'][q].mean():.3f}  {tf[q].mean():.3f}")
    win = {m: 0 for m in OTHER}
    for aM in range(2, 10):
        for aS in range(2, 10):
            q = (d["aM"] == aM) & (d["aS"] == aS)
            if aM != aS and q.sum() >= 300:
                vals = [single[m][q].mean() for m in OTHER]; win[OTHER[int(np.argmax(vals))]] += 1
    print("  best single part per cell (a_M, a_S), cells won:", win)
