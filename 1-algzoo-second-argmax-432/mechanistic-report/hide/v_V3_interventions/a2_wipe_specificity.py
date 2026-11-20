"""Attack on I1, part 2: is 'answers the max' specific to the comparators?  Seeds 5520, 5521 (+5530 for the subset scan).

 1. scale the comparators by c instead of 0
 2. wipe single units, pairs / triples of comparators
 3. wipe every one of the C(16,4) = 1820 sets of four units
"""
from itertools import combinations

from v3lib import *

out = Tee("a2_wipe_specificity.txt")
x, pM, pS, M, S, nb = gather_I1(500_000, 5520)
out(f"seeds 5520..{5520 + nb - 1}, I1-selected n = {len(x)}")
KS = range(1, 7)


def table(fn, xs=x, pMs=pM, pSs=pS, ks=KS, tmax=9):
    """P(ans = pM), P(ans = runner-up) pooled over valid t0 <= tmax, for each k."""
    res = []
    for k in ks:
        v = pMs + k <= tmax
        a = answer(run_iv(xs[v], pMs[v] + k, fn))
        res.append((np.mean(a == pMs[v]), np.mean(a == pSs[v])))
    return np.array(res)


def show(label, r):
    out(f"  {label:34s} P(pM): " + " ".join(f"{v:.3f}" for v in r[:, 0]) + "   P(runner-up): " + " ".join(f"{v:.3f}" for v in r[:, 1]))


out("\n[1] scale h_t0[COMP] by c; k = 1..6; pooled over valid t0 <= 9 (lead's pooling) and over t0 <= 7")
for c in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0):
    show(f"c = {c:<4} (t0 <= 9)", table(scale_units(COMP, c)))
for c in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0):
    show(f"c = {c:<4} (t0 <= 7)", table(scale_units(COMP, c), tmax=7))

out("\n[2] wipe single units / subsets of the comparators; pooled over t0 <= 7 (so that at least two steps follow)")
for u in range(N):
    grp = "COMP" if u in COMP else "PULSE" if u in PULSE else "wave"
    show(f"unit {u} ({grp})", table(scale_units([u], 0.0), tmax=7))
for r in (2, 3):
    for s in combinations(COMP, r):
        show(f"comparators {s}", table(scale_units(list(s), 0.0), tmax=7))
show("comparators (1,6,7,8)", table(scale_units(COMP, 0.0), tmax=7))
show("pulses (4,5,9)", table(scale_units(PULSE, 0.0), tmax=7))
show("all 9 wave units", table(scale_units(WAVE, 0.0), tmax=7))
show("wave units except unit 2", table(scale_units([u for u in WAVE if u != 2], 0.0), tmax=7))
show("everything except comparators", table(scale_units(WAVE + PULSE, 0.0), tmax=7))
show("comparators + pulses", table(scale_units(COMP + PULSE, 0.0), tmax=7))

out("\n[3] every set of four units (1820 sets), 40k selected sequences (seed 5530), pooled over valid t0 <= 9 as the lead")
xs, pMs, pSs, _, _, _ = gather_I1(40_000, 5530, batch=60_000)
xs, pMs, pSs = xs[:40_000], pMs[:40_000], pSs[:40_000]
H0 = run(xs)
sets = list(combinations(range(N), 4))
res = np.zeros((len(sets), len(KS), 2))
for i, s in enumerate(sets):
    res[i] = table(scale_units(list(s), 0.0), xs, pMs, pSs)
ncomp = np.array([len(set(s) & set(COMP)) for s in sets])
np.save(HERE / "a2_subsets.npy", res)
# size of each wipe: |removed part| / |h_t0| averaged over states at t0 = pM + 2
v = pMs + 2 <= 9
h_t0 = H0[v, (pMs + 2)[v]]
relsize = np.array([np.mean(np.linalg.norm(h_t0[:, list(s)], axis=1) / np.linalg.norm(h_t0, axis=1)) for s in sets])
out("  number of comparators in the set -> number of sets; P(ans = pM) pooled over t0: min / median / max over sets, per k = 1..6")
for c in range(5):
    m = ncomp == c
    out(f"  {c} comparators ({m.sum():4d} sets): " + "  ".join(
        f"k={k}: {res[m, j, 0].min():.3f}/{np.median(res[m, j, 0]):.3f}/{res[m, j, 0].max():.3f}" for j, k in enumerate(KS)))
out("  same for P(ans = true runner-up):")
for c in range(5):
    m = ncomp == c
    out(f"  {c} comparators ({m.sum():4d} sets): " + "  ".join(
        f"k={k}: {res[m, j, 1].min():.3f}/{np.median(res[m, j, 1]):.3f}/{res[m, j, 1].max():.3f}" for j, k in enumerate(KS)))
mean_pM = res[:, :, 0].mean(1)
order = np.argsort(-mean_pM)
out("  top 25 sets by mean over k of P(ans = pM):  set | P(pM) k=1..6 | relative size of the wipe (k = 2)")
for i in order[:25]:
    out(f"    {sets[i]}  " + " ".join(f"{v:.3f}" for v in res[i, :, 0]) + f"   {relsize[i]:.3f}")
i_comp = sets.index(tuple(COMP)); i_ctrl = sets.index((10, 11, 12, 15))
out(f"  the comparators {sets[i_comp]}: " + " ".join(f"{v:.3f}" for v in res[i_comp, :, 0]) + f"  rel. size {relsize[i_comp]:.3f}; rank {np.flatnonzero(order == i_comp)[0] + 1} of 1820")
out(f"  lead's control {sets[i_ctrl]}: " + " ".join(f"{v:.3f}" for v in res[i_ctrl, :, 0]) + f"  rel. size {relsize[i_ctrl]:.3f}")
out(f"  best set WITHOUT any comparator: {sets[order[ncomp[order] == 0][0]]} " + " ".join(f"{v:.3f}" for v in res[order[ncomp[order] == 0][0], :, 0]))
out(f"  sets without comparators that are at least as large a wipe as the comparators (rel. size >= {relsize[i_comp]:.3f}): "
    f"{np.sum((ncomp == 0) & (relsize >= relsize[i_comp]))}; their max over sets of mean-over-k P(pM) = "
    f"{mean_pM[(ncomp == 0) & (relsize >= relsize[i_comp])].max() if np.any((ncomp == 0) & (relsize >= relsize[i_comp])) else float('nan'):.3f}")
out(f"  correlation over the 495 comparator-free sets between relative size and mean P(pM): {np.corrcoef(relsize[ncomp == 0], mean_pM[ncomp == 0])[0, 1]:.3f}")
