"""I5. Old events: accuracy and mean margin by (a1, a2) = (age of the max, age of the runner-up) at the last step.
Also the same restricted to 'easy' sequences (T > 0, both gaps >= 0.2 M) to remove the near-tie errors,
and the misfire rate of unit 4 on the runner-up (threshold sag) by (a1 at the time the runner-up arrived)."""
from icommon import *

Z = np.load(HIDE / "r1_I_errors" / "i01_census.npz")
pM, pS, pT, ok, rank, npos = (Z[k] for k in ("pM", "pS", "pT", "ok", "rank", "npos"))
M, S, Tv, margin = (Z[k].astype(np.float64) for k in ("M", "S", "T", "margin"))
a1, a2 = 9 - pM, 9 - pS
n = len(ok)


def table(val, sel=None, fmt="{:.3f}", agg=np.mean, minn=200):
    sel = np.ones(n, bool) if sel is None else sel
    rows = []
    for i in range(10):
        row = [i]
        for j in range(10):
            s = sel & (a1 == i) & (a2 == j)
            row.append("." if i == j else ("-" if s.sum() < minn else fmt.format(agg(val[s]))))
        s = sel & (a1 == i)
        row.append(fmt.format(agg(val[s])))
        rows.append(row)
    rows.append(["all a1"] + [fmt.format(agg(val[sel & (a2 == j)])) for j in range(10)] + [fmt.format(agg(val[sel]))])
    return fmt_table(rows, ["a1 \\ a2"] + [str(j) for j in range(10)] + ["all a2"])


print(f"n = {n}")
print("\n## accuracy by a1 (rows, age of the max) and a2 (columns, age of the runner-up)")
print(table(ok.astype(float)))
print("\n## mean margin logit[pS] - max other logit")
print(table(margin, fmt="{:.2f}"))
print("\n## median margin")
print(table(margin, fmt="{:.2f}", agg=np.median))
print("\n## share of errors that answer the max")
print(table((rank == 1).astype(float), sel=~ok, fmt="{:.2f}", minn=100))
print("\n## share of errors that answer the third")
print(table((rank == 3).astype(float), sel=~ok, fmt="{:.2f}", minn=100))

easy = (Tv > 0) & (M - S >= 0.2 * M) & (S - Tv >= 0.2 * M)
print(f"\n## 'easy' sequences: T > 0, M - S >= 0.2 M, S - T >= 0.2 M: share {easy.mean():.4f}, accuracy {ok[easy].mean():.5f}")
print("error rate (%) by (a1, a2) on easy sequences")
print(table(100 * (~ok).astype(float), sel=easy, fmt="{:.2f}", minn=200))
print("\nmean margin on easy sequences, divided by M (scale-free)")
print(table(margin / M, sel=easy, fmt="{:.2f}", minn=200))

# separate the effects: additive fit of error rate in (a1, a2)? Report main effects and the largest interaction cells
E = np.full((10, 10), np.nan); C = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if i != j:
            s = (a1 == i) & (a2 == j); E[i, j] = 1 - ok[s].mean(); C[i, j] = s.sum()
mu = np.nanmean(E); r = np.nanmean(E, 1) - mu; c = np.nanmean(E, 0) - mu
res = E - (mu + r[:, None] + c[None, :])
print("\n## additive decomposition of the error rate E(a1, a2) = mu + r(a1) + c(a2) + residual (unweighted cell means)")
print(f"mu = {mu:.4f}")
print("r(a1) =", np.round(r, 4))
print("c(a2) =", np.round(c, 4))
tot = np.nansum((E - mu) ** 2)
print(f"share of between-cell variance: a1 main effect {np.nansum((r[:, None] + 0*E) ** 2) / tot:.3f}, a2 main effect {np.nansum((c[None, :] + 0*E) ** 2) / tot:.3f}, interaction {np.nansum(res ** 2) / tot:.3f}")
idx = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(res), axis=None), res.shape))[0][:8]
print("largest positive residuals (a1, a2, error rate, residual):", [(int(i), int(j), round(float(E[i, j]), 3), round(float(res[i, j]), 3)) for i, j in idx])
