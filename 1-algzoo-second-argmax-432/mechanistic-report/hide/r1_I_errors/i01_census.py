"""I1. Error census on 2^21 N(0,1) sequences (seeds 1000..1007, 2^18 each)."""
from icommon import *

keys = ("pM", "pS", "pT", "M", "S", "T", "ans", "rank", "margin", "ok", "npos")
D = {k: [] for k in keys}
for s, x in batches():
    d = describe(x)
    for k in keys:
        D[k].append(d[k])
D = {k: np.concatenate(v) for k, v in D.items()}
n = len(D["ok"]); ok = D["ok"]; err = ~ok
acc = ok.mean()
print(f"n = {n} (2^{np.log2(n):.0f}), accuracy = {acc:.5f} +- {np.sqrt(acc*(1-acc)/n):.5f}, errors = {err.sum()}")
per_seed = ok.reshape(len(SEEDS), -1).mean(1)
print("per-seed accuracy:", np.round(per_seed, 4))

# ---- what is answered instead
print("\n## rank of the answered input among the 10 (2 = correct)")
rows = []
for rk in range(1, 11):
    c = (D["rank"] == rk).sum()
    rows.append([rk, c, f"{c/n:.5f}", f"{(c/err.sum() if rk != 2 else float('nan')):.4f}"])
print(fmt_table(rows, ["rank of answer", "count", "share of all", "share of errors"]))

print("\n## errors: answered position minus true position (ans - pS)")
rows = []
for dlt in range(-9, 10):
    if dlt == 0: continue
    sel = err & (D["ans"] - D["pS"] == dlt)
    # baseline: how often a uniformly random wrong position would have this offset
    rows.append([dlt, sel.sum(), f"{sel.sum()/err.sum():.4f}",
                 f"{(sel & (D['rank']==1)).sum()/max(sel.sum(),1):.3f}",
                 f"{(sel & (D['rank']==3)).sum()/max(sel.sum(),1):.3f}"])
print(fmt_table(rows, ["ans - pS", "count", "share of errors", "of which max", "of which third"]))

print("\n## errors: answer relative to pM (ans - pM), only errors where the answer is not the max")
rows = []
sel0 = err & (D["rank"] != 1)
for dlt in range(-9, 10):
    sel = sel0 & (D["ans"] - D["pM"] == dlt)
    rows.append([dlt, sel.sum(), f"{sel.sum()/sel0.sum():.4f}"])
print(fmt_table(rows, ["ans - pM", "count", "share"]))

print("\n## errors by answered rank and adjacency |ans - pS| = 1")
adj = np.abs(D["ans"] - D["pS"]) == 1
for name, sel in (("max", D["rank"] == 1), ("third", D["rank"] == 3), ("4th+", D["rank"] >= 4)):
    s = err & sel
    print(f"  answered {name}: {s.sum()} ({s.sum()/err.sum():.4f} of errors), adjacent to pS: {(s & adj).mean()/s.mean():.3f}")
# baseline: how often the max / the third is adjacent to the runner-up over all sequences (0.2 for iid inputs)
print(f"  baseline over all sequences: P(|pM - pS| = 1) = {(np.abs(D['pM'] - D['pS']) == 1).mean():.4f},  P(|pT - pS| = 1) = {(np.abs(D['pT'] - D['pS']) == 1).mean():.4f}")
# answered position relative to the max, for errors that answer 4th or lower
s4 = err & (D["rank"] >= 4)
print("  errors answering 4th+: answer - pM distribution: " + "  ".join(f"{dl:+d}:{np.mean(D['ans'][s4] - D['pM'][s4] == dl):.3f}" for dl in range(-9, 10) if dl != 0))
print(f"  errors answering 4th+: share with S <= 0: {np.mean(D['S'][s4] <= 0):.3f}; share at answer = pM - 1: {np.mean(D['ans'][s4] == D['pM'][s4] - 1):.3f}; answer = 0: {np.mean(D['ans'][s4] == 0):.3f}; answer = 9: {np.mean(D['ans'][s4] == 9):.3f}")

# ---- by gaps
gM = D["M"] - D["S"]; gS = D["S"] - D["T"]
edges = np.array([0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, np.inf])
lab = [f"[{edges[i]:g},{edges[i+1]:g})" for i in range(len(edges) - 1)]
bM = np.digitize(gM, edges) - 1; bS = np.digitize(gS, edges) - 1
print("\n## error rate by gaps: rows M-S, columns S-T  (error rate; count in the cell)")
rows = []
for i in range(len(lab)):
    row = [lab[i]]
    for j in range(len(lab)):
        sel = (bM == i) & (bS == j)
        row.append(f"{err[sel].mean():.3f} ({sel.sum()})")
    sel = bM == i
    row.append(f"{err[sel].mean():.4f} ({sel.sum()})")
    rows.append(row)
row = ["all M-S"]
for j in range(len(lab)):
    sel = bS == j
    row.append(f"{err[sel].mean():.4f} ({sel.sum()})")
row.append(f"{err.mean():.4f}")
rows.append(row)
print(fmt_table(rows, ["M-S \\ S-T"] + lab + ["all S-T"]))

print("\n## what is answered, by gap cell: share of the cell's errors that answer max / third / 4th+")
rows = []
for i in range(len(lab)):
    row = [lab[i]]
    for j in range(len(lab)):
        sel = (bM == i) & (bS == j) & err
        if sel.sum() < 30:
            row.append("-")
        else:
            rk = D["rank"][sel]
            row.append(f"{(rk==1).mean():.2f}/{(rk==3).mean():.2f}/{(rk>=4).mean():.2f}")
    rows.append(row)
print(fmt_table(rows, ["M-S \\ S-T"] + lab))

print("\n## share of all errors by gap class (thresholds 0.1)")
for name, sel in (("M-S<0.1 only", (gM < 0.1) & (gS >= 0.1)), ("S-T<0.1 only", (gM >= 0.1) & (gS < 0.1)),
                  ("both<0.1", (gM < 0.1) & (gS < 0.1)), ("both>=0.1", (gM >= 0.1) & (gS >= 0.1)),
                  ("both>=0.2", (gM >= 0.2) & (gS >= 0.2)), ("both>=0.4", (gM >= 0.4) & (gS >= 0.4))):
    print(f"  {name:14s}: share of sequences {sel.mean():.4f}, error rate {err[sel].mean():.4f}, share of errors {(err & sel).sum()/err.sum():.4f}")

# ---- by positions
print("\n## accuracy by (pM rows, pS columns)")
rows = []
for i in range(10):
    row = [i]
    for j in range(10):
        if i == j: row.append("."); continue
        sel = (D["pM"] == i) & (D["pS"] == j)
        row.append(f"{ok[sel].mean():.3f}")
    row.append(f"{ok[D['pM']==i].mean():.4f}")
    rows.append(row)
rows.append(["all pM"] + [f"{ok[D['pS']==j].mean():.4f}" for j in range(10)] + [f"{acc:.4f}"])
print(fmt_table(rows, ["pM \\ pS"] + [str(j) for j in range(10)] + ["all pS"]))
cnt = [(D["pM"] == i) & (D["pS"] == j) for i in range(10) for j in range(10) if i != j]
print("cell counts: min", min(c.sum() for c in cnt), "max", max(c.sum() for c in cnt))

# ---- by number of positive inputs
print("\n## by the number of positive inputs")
rows = []
for k, sel in (("0", D["npos"] == 0), ("1", D["npos"] == 1), ("2", D["npos"] == 2), ("3+", D["npos"] >= 3)):
    e = err & sel
    rk = D["rank"][e]
    rows.append([k, sel.sum(), f"{sel.mean():.5f}", f"{ok[sel].mean():.4f}", f"{e.sum()/err.sum():.4f}",
                 f"{(rk==1).mean():.2f}/{(rk==3).mean():.2f}/{(rk>=4).mean():.2f}"])
print(fmt_table(rows, ["# positive", "count", "share", "accuracy", "share of errors", "errors: max/third/4th+"]))

# ---- before / after
print("\n## by order of arrival of runner-up and max")
rows = []
for k, sel in (("runner-up before max (pS < pM)", D["pS"] < D["pM"]), ("runner-up after max (pS > pM)", D["pS"] > D["pM"])):
    e = err & sel; rk = D["rank"][e]
    rows.append([k, sel.sum(), f"{ok[sel].mean():.4f}", f"{e.sum()/err.sum():.4f}",
                 f"{(rk==1).mean():.3f}/{(rk==3).mean():.3f}/{(rk>=4).mean():.3f}"])
print(fmt_table(rows, ["order", "count", "accuracy", "share of errors", "errors: max/third/4th+"]))
print("\n## by order of arrival of runner-up and third")
rows = []
for k, sel in (("third before runner-up (pT < pS)", D["pT"] < D["pS"]), ("third after runner-up (pT > pS)", D["pT"] > D["pS"])):
    e = err & sel; rk = D["rank"][e]
    rows.append([k, sel.sum(), f"{ok[sel].mean():.4f}", f"{e.sum()/err.sum():.4f}",
                 f"{(rk==1).mean():.3f}/{(rk==3).mean():.3f}/{(rk>=4).mean():.3f}"])
print(fmt_table(rows, ["order", "count", "accuracy", "share of errors", "errors: max/third/4th+"]))
print("\n## by the order of the three (earliest -> latest)")
rows = []
import itertools
for perm in itertools.permutations("MST"):
    pos = {"M": D["pM"], "S": D["pS"], "T": D["pT"]}
    sel = (pos[perm[0]] < pos[perm[1]]) & (pos[perm[1]] < pos[perm[2]])
    e = err & sel; rk = D["rank"][e]
    rows.append(["".join(perm), sel.sum(), f"{ok[sel].mean():.4f}", f"{e.sum()/err.sum():.4f}",
                 f"{(rk==1).mean():.3f}/{(rk==3).mean():.3f}/{(rk>=4).mean():.3f}"])
print(fmt_table(rows, ["arrival order", "count", "accuracy", "share of errors", "errors: max/third/4th+"]))

# ---- by ages
print("\n## by age of the runner-up a2 = 9 - pS and of the max a1 = 9 - pM (marginals)")
rows = []
for a in range(10):
    s2 = D["pS"] == 9 - a; s1 = D["pM"] == 9 - a
    e2 = err & s2; e1 = err & s1
    r2 = D["rank"][e2]; r1 = D["rank"][e1]
    rows.append([a, f"{ok[s2].mean():.4f}", f"{(r2==1).mean():.2f}/{(r2==3).mean():.2f}/{(r2>=4).mean():.2f}",
                 f"{ok[s1].mean():.4f}", f"{(r1==1).mean():.2f}/{(r1==3).mean():.2f}/{(r1>=4).mean():.2f}"])
print(fmt_table(rows, ["age", "acc by a2 (runner-up age)", "errors max/third/4th+", "acc by a1 (max age)", "errors max/third/4th+"]))

np.savez_compressed(HIDE / "r1_I_errors" / "i01_census.npz", **{k: D[k] for k in ("pM", "pS", "pT", "ans", "rank", "ok", "npos")},
                    M=D["M"].astype(np.float32), S=D["S"].astype(np.float32), T=D["T"].astype(np.float32),
                    margin=D["margin"].astype(np.float32))
