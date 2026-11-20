"""Claim I1 (wipe), own reproduction.  Seeds 5300..5304 (5 x 400k sequences).

Selection: second largest of the ten inputs > 0.2 and largest - second > 0.2.
t0 = pM + k <= 9: after computing h_t0 set h_t0[units] = 0, continue with the real inputs.
"""
from v3lib import *

out = Tee("i1_wipe.txt")
SEEDS = range(5300, 5305)
KS = range(1, 10)
SETS = {"COMP [1,6,7,8]": COMP, "control [10,11,12,15]": [10, 11, 12, 15]}
# counts[set][k, t0] = (n, answer = pM, answer = true runner-up)
cnt = {s: np.zeros((10, 10, 3)) for s in SETS}
base = np.zeros(3)                                    # unperturbed: n, ans = pM, ans = pS
nsel = ntot = 0
for seed in SEEDS:
    x = sample(400_000, seed)
    ok, pM, pS, M, S = select_I1(x)
    ntot += len(x); nsel += ok.sum()
    x, pM, pS = x[ok], pM[ok], pS[ok]
    a = answer(run(x)[:, -1])
    base += [len(x), (a == pM).sum(), (a == pS).sum()]
    for name, units in SETS.items():
        for k in KS:
            v = pM + k <= 9
            t0 = pM[v] + k
            a = answer(run_iv(x[v], t0, scale_units(units, 0.0)))
            for t in range(10):
                m = t0 == t
                cnt[name][k, t] += [m.sum(), (a[m] == pM[v][m]).sum(), (a[m] == pS[v][m]).sum()]

out(f"seeds {SEEDS.start}..{SEEDS.stop - 1}; sequences {ntot}; selected {nsel} ({nsel / ntot:.4f})")
out(f"unperturbed on the selection: P(ans = pM) = {base[1] / base[0]:.4f}, P(ans = runner-up) = {base[2] / base[0]:.4f}")
for name in SETS:
    c = cnt[name]
    out(f"\n=== wipe {name} ===")
    for lab, j in (("P(answer = pM)", 1), ("P(answer = true runner-up)", 2)):
        out(f"{lab}; rows k = 1..9, columns t0 = 1..9, last column pooled over valid t0")
        out("  k   " + "".join(f"t0={t:<5d}" for t in range(1, 10)) + " pooled   n_pooled")
        for k in KS:
            row = "".join(f"{c[k, t, j] / c[k, t, 0]:<8.4f}" if c[k, t, 0] > 0 else "   -    " for t in range(1, 10))
            out(f"  {k}   {row} {c[k, :, j].sum() / c[k, :, 0].sum():.4f}   {int(c[k, :, 0].sum())}")
    out("n per (k, t0=9):", [int(c[k, 9, 0]) for k in KS])
    out("pooled over t0 <= 8 only, P(ans = pM), k = 1..8:",
        [round(c[k, :9, 1].sum() / c[k, :9, 0].sum(), 4) for k in range(1, 9)])
    out("pooled over t0 <= 7 only, P(ans = pM), k = 1..7:",
        [round(c[k, :8, 1].sum() / c[k, :8, 0].sum(), 4) for k in range(1, 8)])
    out("pooled over t0 <= 7 only, P(ans = runner-up), k = 1..7:",
        [round(c[k, :8, 2].sum() / c[k, :8, 0].sum(), 4) for k in range(1, 8)])
