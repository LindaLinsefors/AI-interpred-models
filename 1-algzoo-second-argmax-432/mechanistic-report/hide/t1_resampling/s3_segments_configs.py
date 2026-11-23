"""S3. (A) Which inputs can be resampled?  Segments: OLD = inputs older than both maximum and runner-up, MID = between
them, NEW = younger than both.  With three transplants for reference: the resampled state's tape part, its rest, or
the whole resampled state put in place of the original's, and the accuracy of the tape hybrid.
(B) The main candidates by configuration.   Population as S1 (40,000 sequences, seed 3, K = 3).
Run:  python -B s3_segments_configs.py > s3_segments_configs.txt"""
from rs import *

P = Pop(40_000, seed=3)
acc0 = 100 * np.mean(P.ans == P.truth)
xv = np.where(P.real, P.xr, -np.inf)
two = np.argsort(-xv, 1)[:, :2]
lo, hi = two.min(1)[:, None], two.max(1)[:, None]
n_old, n_mid, n_new = (P.real & (COL < lo)).sum(1), ((COL > lo) & (COL < hi)).sum(1), (COL > hi).sum(1)
print(f"{P.n} quiet states from 40000 sequences (steps 3..9); the network answers correctly in {acc0:.2f}%.")
print(f"mean number of inputs per state: OLD {n_old.mean():.2f}, MID {n_mid.mean():.2f}, NEW {n_new.mean():.2f} (+ maximum and runner-up)\n")

cands = [
    ("V0 = max and runner-up (ages, S/M)", V(P)),
    ("V0 + all OLD inputs", V(P, r_seg("old", 10))),
    ("V0 + all MID inputs", V(P, r_seg("mid", 10))),
    ("V0 + all NEW inputs", V(P, r_seg("new", 10))),
    ("V0 + all OLD + all MID", V(P, r_seg("old", 10), r_seg("mid", 10))),
    ("V0 + all OLD + all NEW", V(P, r_seg("old", 10), r_seg("new", 10))),
    ("V0 + all MID + all NEW", V(P, r_seg("mid", 10), r_seg("new", 10))),
    ("V0 + largest OLD", V(P, r_seg("old", 1))),
    ("V0 + largest MID", V(P, r_seg("mid", 1))),
    ("V0 + largest NEW", V(P, r_seg("new", 1))),
    ("V0 + largest of each segment", V(P, r_seg("old", 1), r_seg("mid", 1), r_seg("new", 1))),
    ("V0 + two largest of each segment", V(P, r_seg("old", 2), r_seg("mid", 2), r_seg("new", 2))),
    ("V1 = V0 + third largest", V(P, r_top(3))),
    ("V2 = V1 + fourth largest", V(P, r_top(4))),
    ("V4z = records above zero (top 2 on arrival)", V(P, r_records(2))),
    ("inputs > 0.75 x runner-up on arrival", V(P, r_runthr(0.75))),
    ("top 3 on arrival (above zero)", V(P, r_records(3))),
    ("V3(0) = all positive inputs", V(P, r_above(0.0))),
    ("top 6", V(P, r_top(6))),
]
print("A.  'tape' / 'rest' / 'whole': share (%) of states whose answer changes when that part comes from the resampled state.")
print(f"    'acc': share of tape hybrids that answer correctly (unswapped: {acc0:.2f}).")
print(f"{'candidate V':46s} {'kept':>5s} | FU sum  FU score |   tape   rest  whole |   acc   | clipped  neg %")
res = {}
for name, s in cands:
    r = res[name] = evaluate(P, s, K=3, extra=True)
    m = summary(P, r)
    print(f"{name:46s} {m['nfix']:5.2f} | {m['fu_sum']:6.3f}  {m['fu_sc']:6.3f}   | {m['chg']:6.2f} {100 * r['rest'].mean():6.2f} {100 * r['whole'].mean():6.2f} |"
          f" {100 * r['acc'].mean():6.2f}  | {m['clip']:6.2f}  {m['neg']:5.1f}", flush=True)

print("\nB.  By configuration.  FU sum = E[Var(c | V)] in that group / Var(c) of ALL quiet states (so groups can be compared);")
print("    FU own = the same numerator / the summed variance of c within the group.  'tape' = answer changed %, tape transplant.")
names = ["V0 = max and runner-up (ages, S/M)", "V1 = V0 + third largest", "V4z = records above zero (top 2 on arrival)",
         "inputs > 0.75 x runner-up on arrival"]
vc = P.c.var(0).sum()
groups = [("all", np.ones(P.n, bool)), ("runner-up older than max (a_S > a_M)", P.aS > P.aM), ("runner-up younger than max (a_S < a_M)", P.aS < P.aM)]
groups += [(f"step t = {t}", P.t == t) for t in range(3, 10)]
groups += [(f"a_M = {a}", P.aM == a) for a in range(2, 10)] + [(f"a_S = {a}", P.aS == a) for a in range(2, 10)]
groups += [(f"S/M in ({a:.1f}, {b:.1f}]", (P.S / P.M > a) & (P.S / P.M <= b)) for a, b in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1))]
print(f"{'group':40s} {'n':>7s} {'sd c/M':>7s} |" + "|".join(f"{nm[:26]:^28s}" for nm in names))
print(f"{'':56s} |" + "|".join(f"{'FU sum FU own  tape':^28s}" for _ in names))
for gname, q in groups:
    own = P.c[q].var(0).sum()
    row = f"{gname:40s} {q.sum():7d} {np.sqrt(own):7.2f} |"
    for nm in names:
        w = res[nm]["wc"][q].mean(0).sum()
        row += f"  {w / vc:6.3f} {w / own:6.3f} {100 * res[nm]['chg'][q].mean():6.2f}    |"
    print(row)

print("\nC.  V0: FU sum (relative to all quiet states) for each pair of ages (rows a_M, columns a_S), states at all steps")
w = res[names[0]]["wc"].sum(1) / vc
print("a_M\\a_S " + " ".join(f"{a:6d}" for a in range(2, 10)))
for am in range(2, 10):
    print(f"{am:6d}  " + " ".join(f"{w[(P.aM == am) & (P.aS == a)].mean():6.3f}" if ((P.aM == am) & (P.aS == a)).sum() > 200 else "     ." for a in range(2, 10)))
print("same for the answer-change rate (%) of the tape transplant")
for am in range(2, 10):
    print(f"{am:6d}  " + " ".join(f"{100 * res[names[0]]['chg'][(P.aM == am) & (P.aS == a)].mean():6.2f}" if ((P.aM == am) & (P.aS == a)).sum() > 200 else "     ." for a in range(2, 10)))
