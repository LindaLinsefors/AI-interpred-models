"""I1 (addition). The net is positively homogeneous, so the scale-free gaps are the natural variables:
  g1 = (M - S+)/M,  g2 = (S+ - T+)/M   with  S+ = max(S, 0), T+ = max(T, 0)   (M > 0)
Error rate and share of all errors by (g1, g2) for sequences with S > 0; and a partition of all errors."""
from icommon import *

Z = np.load(HIDE / "r1_I_errors" / "i01_census.npz")
pM, pS, pT, ok, rank, npos = (Z[k] for k in ("pM", "pS", "pT", "ok", "rank", "npos"))
M, S, Tv = (Z[k].astype(np.float64) for k in ("M", "S", "T"))
n = len(ok); err = ~ok; nerr = err.sum()
pos2 = S > 0
g1 = (M - np.maximum(S, 0)) / np.where(M > 0, M, np.nan)
g2 = (np.maximum(S, 0) - np.maximum(Tv, 0)) / np.where(M > 0, M, np.nan)
edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0000001])
lab = [f"[{edges[i]:g},{min(edges[i+1],1):g})" for i in range(len(edges) - 1)]
b1 = np.digitize(g1, edges) - 1; b2 = np.digitize(g2, edges) - 1

print(f"n = {n}, errors = {nerr}; sequences with S > 0: {pos2.mean():.4f}, their error rate {err[pos2].mean():.4f}, share of all errors {(err & pos2).sum()/nerr:.4f}")
print("\n## error rate by relative gaps, S > 0 only: rows g1 = (M-S)/M, columns g2 = (S-T+)/M   [error rate (share of ALL errors in %)]")
rows = []
for i in range(len(lab)):
    row = [lab[i]]
    for j in range(len(lab)):
        sel = pos2 & (b1 == i) & (b2 == j)
        row.append("-" if sel.sum() < 200 else f"{err[sel].mean():.3f} ({100*(err & sel).sum()/nerr:.1f})")
    sel = pos2 & (b1 == i)
    row.append(f"{err[sel].mean():.4f} ({100*(err & sel).sum()/nerr:.1f})")
    rows.append(row)
row = ["all g1"]
for j in range(len(lab)):
    sel = pos2 & (b2 == j)
    row.append(f"{err[sel].mean():.4f} ({100*(err & sel).sum()/nerr:.1f})")
row.append(f"{err[pos2].mean():.4f} ({100*(err & pos2).sum()/nerr:.1f})")
rows.append(row)
print(fmt_table(rows, ["g1 \\ g2"] + lab + ["all g2"]))

print("\n## share of sequences per cell (%), S > 0")
rows = []
for i in range(len(lab)):
    rows.append([lab[i]] + [f"{100*np.mean(pos2 & (b1 == i) & (b2 == j)):.2f}" for j in range(len(lab))])
print(fmt_table(rows, ["g1 \\ g2"] + lab))

print("\n## partition of all errors (mutually exclusive, in this order)")
cls = np.full(n, -1)
names = ["M <= 0 (no positive input)", "S <= 0 < M (one positive input)", "g2 < 0.05 and g1 >= 0.05 (runner-up ~ third)",
         "g1 < 0.05 and g2 >= 0.05 (max ~ runner-up)", "both g1, g2 < 0.05", "0.05 <= min(g1, g2) < 0.1", "0.1 <= min(g1, g2) < 0.2", "both >= 0.2"]
conds = [M <= 0, (M > 0) & (S <= 0), (g2 < 0.05) & (g1 >= 0.05), (g1 < 0.05) & (g2 >= 0.05), (g1 < 0.05) & (g2 < 0.05),
         np.minimum(g1, g2) < 0.1, np.minimum(g1, g2) < 0.2, np.ones(n, bool)]
for k, c in enumerate(conds):
    cls[(cls == -1) & c] = k
rows = []
for k, nm in enumerate(names):
    sel = cls == k; e = err & sel; rk = rank[e]
    rows.append([nm, f"{sel.mean():.4f}", f"{err[sel].mean():.4f}", f"{e.sum()/nerr:.4f}",
                 f"{(rk==1).mean():.2f}/{(rk==3).mean():.2f}/{(rk>=4).mean():.2f}"])
print(fmt_table(rows, ["class", "share of sequences", "error rate", "share of errors", "errors: max/third/4th+"]))

# error rate against g2 alone (fine bins), for g1 >= 0.1 and T > 0; by order of arrival of S and T
print("\n## error rate against g2 = (S - T)/M for T > 0, g1 >= 0.1; by order of arrival and adjacency of S and T")
e2 = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
base = (Tv > 0) & (g1 >= 0.1)
rows = []
for i in range(len(e2) - 1):
    sel = base & (g2 >= e2[i]) & (g2 < e2[i + 1])
    row = [f"[{e2[i]}, {e2[i+1]})", sel.sum(), f"{err[sel].mean():.4f}"]
    for c in (pS < pT, pS > pT, np.abs(pS - pT) == 1, np.abs(pS - pT) > 1, pT == 9, pS == 9):
        s2 = sel & c
        row.append(f"{err[s2].mean():.4f}")
    rows.append(row)
print(fmt_table(rows, ["g2", "n", "all", "S first", "T first", "adjacent", "not adjacent", "T at position 9", "S at position 9"]))
print("\n## error rate against g1 = (M - S)/M for T > 0, g2 >= 0.1; by order of arrival")
rows = []
base = (Tv > 0) & (g2 >= 0.1)
for i in range(len(e2) - 1):
    sel = base & (g1 >= e2[i]) & (g1 < e2[i + 1])
    row = [f"[{e2[i]}, {e2[i+1]})", sel.sum(), f"{err[sel].mean():.4f}"]
    for c in (pM < pS, pM > pS, np.abs(pS - pM) == 1, np.abs(pS - pM) > 1):
        s2 = sel & c
        row.append(f"{err[s2].mean():.4f}")
    row.append(f"{(rank[sel & err] == 1).mean():.3f}")
    rows.append(row)
print(fmt_table(rows, ["g1", "n", "all", "M first", "S first", "adjacent", "not adjacent", "share of errors answering the max"]))
