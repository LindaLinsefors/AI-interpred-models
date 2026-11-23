"""V1. Recomputation of the headline numbers with my own code and a new seed.
Population: quiet states at steps 3..9 of 40,000 N(0,1) sequences (seed 21), K = 3 redraws per state.
FUtape / FUscor / FUrest: unexplained share of variance of the 7 tape coordinates, of the tape piece of the scores
(ages >= 2) and of the 6 coordinates of the five other modes (level, runner, clock, trans, echo); all divided by M.
Transplants: tape = donor's tape part into the recipient; rest = donor's rest with the recipient's tape; whole.
Run:  python -B v1_headline.py > v1_headline.txt"""
from vb import *

P = Pop(40_000, seed=21)
N = P.n
print(f"{N} quiet states from 40000 sequences (seed 21); {int((P.t == 9).sum())} at t = 9; inputs per state {(P.t + 1).mean():.2f}; "
      f"unswapped accuracy {100 * np.mean(P.ans == P.truth):.2f}% (t = 9: {100 * np.mean((P.ans == P.truth)[P.t == 9]):.2f}%)\n")

cands = [
    ("sham (own state)", lambda rng: (P.xr, P.real)),
    ("unrelated quiet state (rescaled to M)", unrelated(P)),
    ("V0 = max and runner-up", V(P)),
    ("top 3", V(P, top(3))), ("top 4", V(P, top(4))), ("top 5", V(P, top(5))), ("top 6", V(P, top(6))),
    ("V0 + D (largest before last record event)", V(P, segment("pre", 1))),
    ("V0 + largest after the last record event", V(P, segment("new", 1))),
    ("contenders c = 1 (records above 0)", V(P, contenders(1.0))),
    ("contenders c = 0.9", V(P, contenders(0.9))),
    ("contenders c = 0.75", V(P, contenders(0.75))),
    ("contenders c = 0.5", V(P, contenders(0.5))),
    ("contenders c = 0.25", V(P, contenders(0.25))),
    ("contenders c = 0 (all positive inputs)", V(P, contenders(0.0))),
    ("inputs > 0.75 S_t (final runner-up)", V(P, above_final(0.75))),
    ("inputs > 0.5 S_t", V(P, above_final(0.5))),
    ("earlier maxima only (above 0)", V(P, arrival(1))),
    ("top 3 on arrival (above 0)", V(P, arrival(3))),
    ("record history, no floor (top 2 on arrival)", V(P, arrival(2, -np.inf))),
    ("V0 + inputs of age 0, 1", V(P, ages([0, 1]))),
    ("V0 + inputs of age 0..3", V(P, ages([0, 1, 2, 3]))),
    ("V0 + input of age 0", V(P, ages([0]))), ("V0 + input of age 5", V(P, ages([5]))), ("V0 + input of age 9", V(P, ages([9]))),
    ("V0 + all OLD", V(P, segment("old"))), ("V0 + all MID", V(P, segment("mid"))), ("V0 + all NEW", V(P, segment("new"))),
    ("V0 + all OLD + all MID", V(P, segment("old"), segment("mid"))),
    ("V0 + all OLD + all NEW", V(P, segment("old"), segment("new"))),
    ("V0 + all MID + all NEW", V(P, segment("mid"), segment("new"))),
]
res = {}
for label, q in (("ALL STEPS 3..9", np.ones(N, bool)), ("t = 9 ONLY (FU still relative to all quiet states)", P.t == 9)):
    print(label); print(HEAD)
    for name, s in cands:
        if name not in res:
            res[name] = evaluate(P, s)
        print(line(P, name, res[name], q), flush=True)
    print()

two = np.sort(np.stack([P.cM, P.cS], 1), 1)
n_old, n_mid, n_new = (P.real & (COL < two[:, :1])).sum(1), ((COL > two[:, :1]) & (COL < two[:, 1:])).sum(1), (COL > two[:, 1:]).sum(1)
print(f"mean inputs per state: older than both {n_old.mean():.2f}, between {n_mid.mean():.2f}, younger than both {n_new.mean():.2f}")
m75 = fixed(P, P.xv, (top(2), contenders(0.75)))[0]
free = P.real & ~m75
print(f"contenders c = 0.75: kept {m75.sum(1).mean():.2f}; share of states with 2, 3, 4, 5+ kept "
      f"{np.round([np.mean(m75.sum(1) == k) for k in (2, 3, 4)] + [np.mean(m75.sum(1) >= 5)], 3)}; "
      f"redrawn inputs {free.sum(1).mean():.2f} per state, {100 * np.mean(P.xr[free] < 0):.1f}% of them negative")
has_d = (P.real & (COL < two[:, 1:]) & (COL != two[:, :1])).any(1)
print(f"D exists in {100 * has_d.mean():.1f}% of states\n")

# ------------------------------------------------------------------ small M, heavy tail
print("FU of the tape coordinates within bins of M_t (numerator and denominator from the bin), and on M_t > 0.5; share of the numerator from the 1% of states with the largest within-V variance")
BINS = ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 99))
print(f"{'V':40s} | " + " ".join(f"({a},{b}]".rjust(9) for a, b in BINS) + " |  M>0.5 | top-1% share | answer changed % (tape) per bin")
for name in ("V0 = max and runner-up", "top 3", "contenders c = 1 (records above 0)", "contenders c = 0.75", "contenders c = 0 (all positive inputs)"):
    r = res[name]; wsum = r["cT"].sum(1)
    fb = [wsum[(P.M > a) & (P.M <= b)].mean() / P.f["cT"][(P.M > a) & (P.M <= b)].var(0).sum() for a, b in BINS]
    cb = [100 * r["tape"][(P.M > a) & (P.M <= b)].mean() for a, b in BINS]
    q5 = P.M > 0.5
    print(f"{name:40s} | " + " ".join(f"{v:9.4f}" for v in fb) + f" | {wsum[q5].mean() / P.f['cT'][q5].var(0).sum():6.4f} |   {np.sort(wsum)[-N // 100:].sum() / wsum.sum():6.3f}     | "
          + " ".join(f"{v:6.2f}" for v in cb))
print("share of states per bin:", np.round([np.mean((P.M > a) & (P.M <= b)) for a, b in BINS], 3), "\n")

# ------------------------------------------------------------------ by configuration
vc = P.f["cT"].var(0).sum()
A, B = "V0 = max and runner-up", "contenders c = 0.75"
groups = [("all", np.ones(N, bool)), ("runner-up older than max", P.aS > P.aM), ("runner-up younger than max", P.aS < P.aM)]
groups += [(f"t = {t}", P.t == t) for t in range(3, 10)] + [(f"a_M = {a}", P.aM == a) for a in range(2, 10)] + [(f"a_S = {a}", P.aS == a) for a in range(2, 10)]
print("By configuration: FU = E[Var(c | V)] in the group / Var(c) of ALL quiet states; 'tape' = answer changed % under the tape transplant")
print(f"{'group':30s} {'n':>7s} | V0: FU   tape% | contenders 0.75: FU   tape%  whole%")
for g, q in groups:
    print(f"{g:30s} {q.sum():7d} | {res[A]['cT'][q].sum(1).mean() / vc:7.3f} {100 * res[A]['tape'][q].mean():6.2f} |"
          f"        {res[B]['cT'][q].sum(1).mean() / vc:10.3f} {100 * res[B]['tape'][q].mean():6.2f} {100 * res[B]['whole'][q].mean():6.2f}")
print("\nV0: FU for each pair of ages (rows a_M, columns a_S), cells with > 200 states")
print("a_M\\a_S " + " ".join(f"{a:6d}" for a in range(2, 10)))
wA = res[A]["cT"].sum(1) / vc
for am in range(2, 10):
    print(f"{am:6d}  " + " ".join(f"{wA[(P.aM == am) & (P.aS == a)].mean():6.3f}" if ((P.aM == am) & (P.aS == a)).sum() > 200 else "     ." for a in range(2, 10)))

xs = np.sort(P.x, 1)
gap = (xs[:, -2] - xs[:, -3]) / xs[:, -1]
print("\nTape transplant, answer changed %, by the gap (S - T)/M of the WHOLE sequence")
for a, b in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 99)):
    q = (gap > a) & (gap <= b)
    print(f"  gap in ({a:.2f}, {b:.2f}]  n = {q.sum():6d}   V0 {100 * res[A]['tape'][q].mean():6.2f}   contenders 0.75: tape {100 * res[B]['tape'][q].mean():6.2f}  "
          f"rest {100 * res[B]['rest'][q].mean():6.2f}  whole {100 * res[B]['whole'][q].mean():6.2f}   unswapped accuracy {100 * np.mean((P.ans == P.truth)[q]):6.2f}")

# ------------------------------------------------------------------ is the dependence on V special to the tape?
print("\nSPECIFICITY.  The same FU for other parts of the state (all / M): the 6 coordinates of the other five modes, the 3 pulse units,")
print("and the score entries of age >= 2 with the mean over the ten entries removed: tape piece, rest piece, full score.")
print(f"{'V':44s} | tape c  rest c  pulses | centred scores: tape   rest   full")
for name, _ in cands[1:]:
    r = res[name]
    print(f"{name:44s} | {fu(P, r, 'cT'):6.3f}  {fu(P, r, 'cR'):6.3f}  {fu(P, r, 'pulse'):6.3f} |                {fu(P, r, 'dT'):6.3f} {fu(P, r, 'dR'):6.3f} {fu(P, r, 'dF'):6.3f}")
print("per rest coordinate (level, runner, clock.re, clock.im, trans, echo):")
for name in (A, "top 3", B, "contenders c = 0 (all positive inputs)"):
    print(f"  {name:42s} " + " ".join(f"{v:6.3f}" for v in fu(P, res[name], "cR", None, True)))
print("sd over all quiet states of c/M: tape", np.round(P.f["cT"].std(0), 2), " rest", np.round(P.f["cR"].std(0), 2))
