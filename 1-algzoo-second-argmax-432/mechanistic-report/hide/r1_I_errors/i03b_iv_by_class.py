"""I3 (detail). Which errors do the pulse interventions repair?  Errors are split by class (sign / relative gaps /
order of arrival) and, for each class, the share that (b1), (c), (d) repair and the new errors they make."""
from icommon import *
from i03_interventions import run_iv

IV = [("b1", {"b1"}), ("c", {"c"}), ("d", {"b1", "b2", "c"})]
acc = {}
names = ["M <= 0", "S <= 0 < M", "S~T (g2<0.05<=g1), S first", "S~T (g2<0.05<=g1), T first",
         "M~S (g1<0.05<=g2), M first", "M~S (g1<0.05<=g2), S first", "both gaps < 0.05", "min gap in [0.05, 0.1)", "min gap >= 0.1"]
tot = {nm: np.zeros(2 + 2 * len(IV)) for nm in names}
ans_tot = {}                       # for S~T classes: answered third before / after
for s, x in batches():
    d = describe(x)
    pM, pS, pT, M, S, Tv, ok = d["pM"], d["pS"], d["pT"], d["M"], d["S"], d["T"], d["ok"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = (M - np.maximum(S, 0)) / M; g2 = (np.maximum(S, 0) - np.maximum(Tv, 0)) / M
    conds = [M <= 0, (M > 0) & (S <= 0),
             (S > 0) & (g2 < 0.05) & (g1 >= 0.05) & (pS < pT), (S > 0) & (g2 < 0.05) & (g1 >= 0.05) & (pS > pT),
             (S > 0) & (g1 < 0.05) & (g2 >= 0.05) & (pM < pS), (S > 0) & (g1 < 0.05) & (g2 >= 0.05) & (pM > pS),
             (S > 0) & (g1 < 0.05) & (g2 < 0.05), (S > 0) & (np.minimum(g1, g2) >= 0.05) & (np.minimum(g1, g2) < 0.1),
             (S > 0) & (np.minimum(g1, g2) >= 0.1)]
    assert np.all(np.sum(conds, 0) == 1)
    oks = {}
    for nm, mode in IV:
        lg, _ = run_iv(x, mode)
        oks[nm] = lg.argmax(1) == pS
    for nm, c in zip(names, conds):
        v = [c.sum(), (c & ~ok).sum()]
        for k, _ in IV:
            v += [(c & ~ok & oks[k]).sum(), (c & ok & ~oks[k]).sum()]
        tot[nm] += v

nerr = sum(v[1] for v in tot.values()); n = sum(v[0] for v in tot.values())
print(f"n = {n}, errors = {nerr}")
rows = []
for nm in names:
    v = tot[nm]
    row = [nm, f"{v[0]/n:.4f}", f"{v[1]/v[0]:.4f}", f"{v[1]/nerr:.4f}"]
    for i in range(len(IV)):
        row.append(f"{v[2+2*i]/v[1]:.3f} / {v[3+2*i]/v[1]:.3f}")
    rows.append(row)
v = sum(tot.values())
rows.append(["all", "1", f"{v[1]/v[0]:.4f}", "1"] + [f"{v[2+2*i]/v[1]:.3f} / {v[3+2*i]/v[1]:.3f}" for i in range(len(IV))])
print(fmt_table(rows, ["class", "share of sequences", "error rate", "share of errors"] +
                [f"({k}): fixed / new, as share of the class's errors" for k, _ in IV]))
