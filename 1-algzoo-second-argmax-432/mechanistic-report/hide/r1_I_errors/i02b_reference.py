"""I2 (explanation).
 (1) Does 'relative gaps shrink' explain the loss under positive shifts and under Unif(-1,1)?  Predict the accuracy of the
     T > 0 class from the N(0,1), c = 0 error table E(g1, g2), g1 = (M-S)/M, g2 = (S-T)/M, by re-weighting.
 (2) The implicit reference: sweep the runner-up through zero (third clearly negative) and find where the net starts to answer it.
 (3) What the net answers when S <= 0.
"""
from icommon import *

# ---------- (1) re-weighting
E1 = np.r_[0, np.geomspace(0.002, 1.0, 24)]            # g bins
def gcell(M, S, Tv):
    g1 = (M - S) / M; g2 = (S - Tv) / M
    return np.clip(np.digitize(g1, E1) - 1, 0, len(E1) - 2), np.clip(np.digitize(g2, E1) - 1, 0, len(E1) - 2)

nb = len(E1) - 1
cnt = np.zeros((nb, nb)); errs = np.zeros((nb, nb))
for s, x in batches():
    d = describe(x)
    sel = d["T"] > 0
    i, j = gcell(d["M"][sel], d["S"][sel], d["T"][sel])
    np.add.at(cnt, (i, j), 1); np.add.at(errs, (i, j), ~d["ok"][sel])
Etab = np.where(cnt > 0, errs / np.maximum(cnt, 1), np.nan)
print(f"(1) error table from N(0,1), T > 0: {int(cnt.sum())} sequences, {nb}x{nb} cells, {int((cnt >= 50).sum())} cells with n >= 50")
rows = []
for name, gen in [(f"N(0,1) + {c}", (lambda s, c=c: sample(BATCH, s + 200) + c)) for c in (0, 0.5, 1, 2, 3)] + \
                 [("Unif(-1,1)", lambda s: sample(BATCH, s + 300, "uniform")), ("Unif(0,1)", lambda s: (sample(BATCH, s + 300, "uniform") + 1) / 2)]:
    a_true = []; a_pred = []; share = []; cov = []
    for s in SEEDS[:4]:
        x = gen(s)
        d = describe(x)
        sel = d["T"] > 0
        i, j = gcell(d["M"][sel], d["S"][sel], d["T"][sel])
        e = Etab[i, j]; okc = cnt[i, j] >= 50
        a_true.append(d["ok"][sel].mean()); a_pred.append(1 - np.nanmean(e[okc])); share.append(sel.mean()); cov.append(okc.mean())
        a_true[-1] = (d["ok"][sel].mean(), d["ok"][sel][okc].mean())
    rows.append([name, f"{np.mean(share):.4f}", f"{np.mean([a[0] for a in a_true]):.4f}", f"{np.mean([a[1] for a in a_true]):.4f}",
                 f"{np.mean(a_pred):.4f}", f"{np.mean(cov):.4f}"])
print(fmt_table(rows, ["inputs (fresh seeds)", "share with T > 0", "accuracy, T > 0", "same, in covered cells", "predicted from E(g1, g2)", "coverage"]))

# ---------- (2) sweep the runner-up through zero
x = sample(2 ** 18, 1060)
pM, pS, pT, M, S, Tv = top3(x)
sel = (M > 0) & (Tv < -0.3 * M)                         # third clearly negative: the runner-up can be moved in [-0.25 M, 0.25 M]
x, pM, pS, M = x[sel], pM[sel], pS[sel], M[sel]
r = np.arange(len(x))
grid = np.round(np.arange(-0.25, 0.2501, 0.01), 3)
P = np.empty((len(x), len(grid)), bool)
for i, g in enumerate(grid):
    xx = x.copy(); xx[r, pS] = g * M
    P[:, i] = logits(run(xx)).argmax(1) == pS
print(f"\n(2) runner-up moved to s*M, third < -0.3 M (n = {len(x)}, seed 1060): P(net answers the runner-up's position)")
print(fmt_table([[f"{g:+.2f}", f"{P[:, i].mean():.4f}"] for i, g in enumerate(grid) if abs(g * 100 % 5) < 1e-9 or abs(g) <= 0.05],
                ["s = S/M", "P(answer = p_S)"]))
first = np.where(P.any(1), grid[P.argmax(1)], np.inf)
print("smallest s on the grid at which the answer is p_S: quantiles 5/25/50/75/95% = " +
      " / ".join(f"{v:+.2f}" for v in np.quantile(first[np.isfinite(first)], [.05, .25, .5, .75, .95])) +
      f";  never within the grid: {np.mean(~np.isfinite(first)):.4f};  already at s = -0.25: {np.mean(first == grid[0]):.4f}")
a1 = 9 - pM; a2 = 9 - pS
print("median of that s by order and adjacency: " + "; ".join(
    f"{nm}: {np.median(first[c & np.isfinite(first)]):+.2f} (n={c.sum()})" for nm, c in
    (("S before M", pS < pM), ("S after M", pS > pM), ("S right after M", pS == pM + 1), ("S right before M", pS == pM - 1), ("S at position 9", pS == 9))))
print("P(answer = p_S) at s = -0.05 / +0.05 / +0.10 by where the runner-up is:")
for nm, c in (("S before M", pS < pM), ("S after M, not adjacent, not last", (pS > pM + 1) & (pS < 9)), ("S right after M", pS == pM + 1), ("S at position 9", pS == 9)):
    print(f"   {nm}: " + " / ".join(f"{P[c, int(np.argmin(np.abs(grid - g)))].mean():.3f}" for g in (-0.05, 0.05, 0.10)))

# ---------- (3) what is answered when S <= 0
print("\n(3) answers when exactly one input is positive (M > 0 >= S), N(0,1), seeds 1000..1007")
off = np.zeros(19); n1 = 0; hit = {k: 0 for k in ("pS", "pM", "pM-1", "pM+1", "9", "pT")}
ans0 = np.zeros(10); n0 = 0; rel0 = {k: 0 for k in ("pM (largest, negative)", "pS", "9", "argmin", "position of the last input above -0.3")}
for s, x in batches():
    d = describe(x)
    c = (d["M"] > 0) & (d["S"] <= 0)
    a = d["ans"][c]; pm = d["pM"][c]
    np.add.at(off, a - pm + 9, 1); n1 += c.sum()
    hit["pS"] += (a == d["pS"][c]).sum(); hit["pM"] += (a == pm).sum(); hit["pM-1"] += (a == pm - 1).sum(); hit["pM+1"] += (a == pm + 1).sum()
    hit["9"] += (a == 9).sum(); hit["pT"] += (a == d["pT"][c]).sum()
    c0 = d["M"] <= 0
    a = d["ans"][c0]; n0 += c0.sum(); np.add.at(ans0, a, 1)
    rel0["pM (largest, negative)"] += (a == d["pM"][c0]).sum(); rel0["pS"] += (a == d["pS"][c0]).sum(); rel0["9"] += (a == 9).sum()
    rel0["argmin"] += (a == x[c0].argmin(1)).sum()
print(f"n = {n1}.  answer - p_M distribution: " + "  ".join(f"{k-9:+d}:{v/n1:.3f}" for k, v in enumerate(off) if v / n1 >= 0.01))
print("   " + "  ".join(f"P(answer = {k}) = {v/n1:.3f}" for k, v in hit.items()))
print(f"no positive input (M <= 0): n = {n0}.  answer distribution over positions: " + " ".join(f"{p}:{v/n0:.3f}" for p, v in enumerate(ans0)))
print("   " + "  ".join(f"P(answer = {k}) = {v/n0:.3f}" for k, v in rel0.items() if v > 0))
