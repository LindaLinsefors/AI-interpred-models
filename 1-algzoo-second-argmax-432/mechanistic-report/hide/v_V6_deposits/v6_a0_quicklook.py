"""V6 quick look (natural samples from d4_seqs.npz): D4 split by which candidate is older, by whether
the runner-up sits right before the max (g = 0), by order class, and the answer distribution."""
from v6_data import *

out = open(HERE / "a0_quicklook.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


d = dict(np.load(HERE / "d4_seqs.npz"))
pM, pS, pT, ans = d["pM"], d["pS"], d["pT"], d["ans"]
rg = (d["S"] - d["T"]) / d["S"]
edges = [0, .03, .08, .2, .5]
D = np.abs(pS - pT)
bins = [(rg > edges[b]) & (rg <= edges[b + 1]) for b in range(4)]


def tab(mask, what=lambda m: ans[m] == pT[m]):
    cells = []
    for b in range(4):
        m = mask & bins[b]
        cells.append(f"{what(m).mean():.3f} ({m.sum():6d})" if m.sum() >= 30 else f"  -   ({m.sum():6d})")
    return "  ".join(cells)


late = (pM > pS) & (pM > pT)
P("P(answer = pT); columns = (S-T)/S in (0,.03] (.03,.08] (.08,.2] (.2,.5];  n in brackets")
for name, cls in (("max LAST  (pM > pS, pT)  [the lead's D4]", late),
                  ("max FIRST (pM < pS, pT)", (pM < pS) & (pM < pT)),
                  ("S < M < T  (third arrives after the max)", (pS < pM) & (pM < pT)),
                  ("T < M < S  (runner-up arrives after the max)", (pT < pM) & (pM < pS))):
    P(f"\n== {name}")
    for dd in range(1, 9):
        P(f"  D={dd}: {tab(cls & (D == dd))}")
    P(f"  all D: {tab(cls)}")

P("\n== max LAST split: runner-up S OLDER than third (pS < pT) | S YOUNGER (pS > pT), not adjacent to max (pS < pM-1) | S right before the max (pS = pM-1, 'g = 0')")
for dd in range(1, 9):
    P(f"  D={dd}: S older: {tab(late & (D == dd) & (pS < pT))}")
    P(f"       S younger, g>=1: {tab(late & (D == dd) & (pS > pT) & (pS < pM - 1))}")
    P(f"       S younger, g=0 : {tab(late & (D == dd) & (pS == pM - 1))}")

P("\n== max LAST, D = 7 and 8 by configuration (pS, pT, pM)")
for cfg in ((0, 8, 9), (8, 0, 9), (0, 7, 8), (7, 0, 8), (0, 7, 9), (7, 0, 9), (1, 8, 9), (8, 1, 9)):
    m = (pS == cfg[0]) & (pT == cfg[1]) & (pM == cfg[2])
    P(f"  (pS, pT, pM) = {cfg}: {tab(m)}")

P("\n== max LAST: what does the network answer?  rows D; per gap bin: P(pS) P(pT) P(pM) P(4th largest) P(other)")
p4 = d["p4"]
for dd in range(1, 9):
    cells = []
    for b in range(4):
        m = late & (D == dd) & bins[b]
        a = ans[m]
        o = [np.mean(a == pS[m]), np.mean(a == pT[m]), np.mean(a == pM[m]), np.mean(a == p4[m])]
        cells.append(" ".join(f"{v:.3f}" for v in o + [1 - sum(o)]))
    P(f"  D={dd}: " + " | ".join(cells))
out.close()
