"""S5. The compact candidates on a fresh sample, with controls.
(A) "Contenders": every input that exceeded c x (the runner-up of the inputs before it, floored at 0) when it arrived,
    for c = 1 (= records above zero, V4z), 0.9, 0.75, 0.5, 0.25, 0; and V0 + D, D = the displaced runner-up = the
    largest input that arrived before the last record event.  All steps and t = 9 alone.
(B) D: its age without its value, its value without its age.
(C) For the c = 0.75 set: what is left, in logits, and which states change their answer.
Population: quiet states at steps 3..9 of 40,000 sequences (seed 5), K = 3.
Run:  python -B s5_final.py > s5_final.txt"""
from rs import *

P = Pop(40_000, seed=5)
N = P.n
acc0 = 100 * np.mean(P.ans == P.truth)
ar = np.arange(N)
xv = np.where(P.real, P.xr, -np.inf)
two = np.argsort(-xv, 1)[:, :2]
lo, hi = two.min(1)[:, None], two.max(1)[:, None]
pre, new = P.real & (COL < hi) & (COL != lo), COL > hi
has = np.flatnonzero(pre.any(1))
dcol = np.where(pre, xv, -np.inf).argmax(1)
tn = lambda rng, b: ndtri(rng.uniform(1e-300, 1, (N, 10)) * ndtr(b)[:, None])


def d_age_only(rng):
    out = np.where(pre | new, tn(rng, P.S), P.xr)                         # every other input iid below S ...
    m = np.where(pre, out, -np.inf).argmax(1)                             # ... and the largest early one moved to D's step
    a, b = out[has, m[has]].copy(), out[has, dcol[has]].copy()
    out[has, m[has]], out[has, dcol[has]] = b, a
    return out, None


def d_value_only(rng):
    D = np.where(pre.any(1), xv[ar, dcol], P.S)
    out = np.where(pre, tn(rng, D), np.where(new, tn(rng, P.S), P.xr))    # early inputs below D, later ones below S ...
    j = np.where(pre, rng.uniform(size=(N, 10)), 9.0).argmin(1)           # ... and D at a random early step
    out[has, j[has]] = D[has]
    return out, None


cands = [
    ("unrelated quiet state (rescaled to M)", lambda rng: (P.xr[rng.permutation(N)], None)),
    ("V0 = max and runner-up (ages, S/M)", V(P)),
    ("V0 + age of D (not its value)", d_age_only),
    ("V0 + value of D (not its age)", d_value_only),
    ("V0 + D (age and value)", V(P, r_seg("pre", 1))),
    ("contenders c = 1 (records above zero)", V(P, r_records(2))),
    ("contenders c = 0.9", V(P, r_runthr(0.9))),
    ("contenders c = 0.75", V(P, r_runthr(0.75))),
    ("contenders c = 0.5", V(P, r_runthr(0.5))),
    ("contenders c = 0.25", V(P, r_runthr(0.25))),
    ("contenders c = 0 (all positive inputs)", V(P, r_runthr(0.0))),
]
print(f"{N} quiet states from 40000 sequences (steps 3..9), {int((P.t == 9).sum())} of them final (t = 9).  Unswapped accuracy {acc0:.2f}%.")
print(f"D exists in {100 * len(has) / N:.1f}% of the states.\n")
head = f"{'V':40s} {'kept':>5s} | " + " ".join(f"{c:>8s}" for c in COORD) + " | FU sum FU score |   tape   rest  whole |   acc  | clipped neg %"
res = {}
for label, q in (("A/B. all steps", np.ones(N, bool)), ("t = 9 only (FU still relative to all quiet states)", P.t == 9)):
    print(label + f"   (unswapped accuracy {100 * np.mean(P.ans[q] == P.truth[q]):.2f}%)")
    print(head)
    for name, s in cands:
        r = res[name] = res.get(name) or evaluate(P, s, K=3, extra=True)
        m = summary(P, r, q)
        print(f"{name:40s} {m['nfix']:5.2f} | " + " ".join(f"{v:8.3f}" for v in m["fu"]) + f" | {m['fu_sum']:6.3f} {m['fu_sc']:6.3f}   |"
              f" {m['chg']:6.2f} {100 * r['rest'][q].mean():6.2f} {100 * r['whole'][q].mean():6.2f} | {100 * r['acc'][q].mean():6.2f} |"
              f" {m['clip']:6.2f} {m['neg']:5.1f}", flush=True)
    print()

# ------------------------------------------------------------------ C. what is left under the c = 0.75 set
name = "contenders c = 0.75"
r = res[name]
mask = fixed(xv, (r_top(2), r_runthr(0.75)))[0]
free = P.real & ~mask
print(f"C. {name}: {mask.sum(1).mean():.2f} inputs kept, {free.sum(1).mean():.2f} resampled per state; share of states with 2, 3, 4, 5+ kept:",
      np.round([np.mean(mask.sum(1) == k) for k in (2, 3, 4)] + [np.mean(mask.sum(1) >= 5)], 3))
print(f"   resampled inputs: {100 * np.mean(P.xr[free] > 0):.1f}% positive, median x/S = {np.median((P.xr / P.S[:, None])[free]):.2f}, "
      f"largest resampled input per state, median x/S = {np.median(np.where(free, P.xr, -np.inf).max(1)[free.any(1)] / P.S[free.any(1)]):.2f}")
print("   residual sd of the tape piece of the scores (in units of M) by age of the entry, against its sd over all quiet states:")
print("     age     " + " ".join(f"{9 - i:6d}" for i in range(8)))
print("     residual" + " ".join(f"{v:6.2f}" for v in np.sqrt(r["wg"].mean(0))))
print("     total   " + " ".join(f"{v:6.2f}" for v in P.g.std(0)))
print("     same for V0:")
print("     residual" + " ".join(f"{v:6.2f}" for v in np.sqrt(res['V0 = max and runner-up (ages, S/M)']["wg"].mean(0))))

# margin of the unswapped answer: top logit minus second logit at the end of the run, in units of the final maximum
def final_logits(h, x, t):
    out = np.empty((len(h), 10))
    for tt in np.unique(t):
        q = t == tt
        out[q] = (h[q] if tt == 9 else continue_from(h[q], x[q, tt + 1:])[:, -1]) @ U.T
    return out
L = np.sort(final_logits(P.h, P.x, P.t), 1)
margin = (L[:, -1] - L[:, -2]) / P.x.max(1)
edges = np.quantile(margin, np.linspace(0, 1, 11))
print("\n   answer-change rate of the tape transplant by decile of the margin of the unswapped answer (top minus second logit, / final max):")
print("     decile upper edge " + " ".join(f"{e:7.1f}" for e in edges[1:]))
for nm in ("V0 = max and runner-up (ages, S/M)", name):
    k = np.clip(np.digitize(margin, edges[1:-1]), 0, 9)
    print(f"     {nm[:14]:14s} %  " + " ".join(f"{100 * res[nm]['chg'][k == j].mean():7.2f}" for j in range(10)))
xs = np.sort(P.x, 1)
gap = (xs[:, -2] - xs[:, -3]) / xs[:, -1]
print("   and by the gap between runner-up and third of the WHOLE sequence, (S - T)/M at the end:")
for a, b in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 9)):
    q = (gap > a) & (gap <= b)
    print(f"     gap in ({a:.2f}, {b:.2f}]  n = {q.sum():6d}   V0: {100 * res['V0 = max and runner-up (ages, S/M)']['chg'][q].mean():6.2f}%   "
          f"{name}: {100 * r['chg'][q].mean():6.2f}%   whole-state reference: {100 * r['whole'][q].mean():6.2f}%   unswapped accuracy {100 * np.mean((P.ans == P.truth)[q]):6.2f}%")
