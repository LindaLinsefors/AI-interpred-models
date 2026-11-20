"""V6 attack: analysis of the per-configuration table a2_configs.npz (see v6_a2_configs.py)."""
from v6_data import *

out = open(HERE / "a2_analyse.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


z = np.load(HERE / "a2_configs.npz")
cnt, nbin, reg, rec, ymed = z["cnt"], z["nbin"], z["reg"], z["rec"], z["ymed"]
B = int(z["B"])
P(f"controlled sampling: seed {int(z['seed'])}, {B} base sequences reused in each of the 720 configurations")
P(f"sequences per gap bin (T > 0.2): {nbin.tolist()}  [bins (0,.03] (.03,.08] (.08,.2] (.2,.5]]")
cfgs = [(pM, pS, pT) for pM in range(10) for pS in range(10) for pT in range(10) if len({pM, pS, pT}) == 3]
errT = lambda c, b: cnt[c[0], c[1], c[2], b, 1] / nbin[b]            # P(answer = pT) in bin b


def klass(c):
    pM, pS, pT = c
    if pM > pS and pM > pT: return "last"
    if pM < pS and pM < pT: return "first"
    return "S<M<T" if pS < pM else "T<M<S"


lead = np.array([[.183, .066, .029, .005], [.147, .038, .010, .003], [.189, .029, .003, .000], [.245, .020, .001, .000],
                 [.276, .028, .002, .000], [.338, .137, .023, .002], [.371, .171, .125, .066], [.447, .346, .123, .004]])
P("\n[1] D4 rebuilt from the configurations (equal weight per configuration) vs the lead")
for k in ("last", "first", "S<M<T", "T<M<S"):
    P(f"  order class {k}:")
    for d in range(1, 9):
        cs = [c for c in cfgs if klass(c) == k and abs(c[1] - c[2]) == d]
        if not cs:
            continue
        row = [np.mean([errT(c, b) for c in cs]) for b in range(4)]
        extra = ("   lead: " + " ".join(f"{v:.3f}" for v in lead[d - 1])) if k == "last" else ""
        P(f"    D={d} ({len(cs):3d} configs): " + " ".join(f"{v:.3f}" for v in row) + extra)

P("\n[2] max-last, each separation: spread of P(answer = pT) over configurations, bin (.08,.2] and bin (.03,.08]")
P("    D  nconf |  bin (.08,.2]: mean   min   max  | bin (.03,.08]: mean   min   max | configs with the largest error in bin (.08,.2] (pS,pT,pM: rate)")
for d in range(1, 9):
    cs = [c for c in cfgs if klass(c) == "last" and abs(c[1] - c[2]) == d]
    e2 = np.array([errT(c, 2) for c in cs]); e1 = np.array([errT(c, 1) for c in cs])
    top = np.argsort(-e2)[:4]
    P(f"    {d}  {len(cs):4d}  |  {e2.mean():.3f} {e2.min():.3f} {e2.max():.3f}  |  {e1.mean():.3f} {e1.min():.3f} {e1.max():.3f} | "
      + ", ".join(f"({cs[i][1]},{cs[i][2]},{cs[i][0]}): {e2[i]:.3f}" for i in top))

P("\n[3] max-last: share of the between-configuration variance of P(answer = pT) explained by one grouping factor (eta^2, 240 configs)")
late = [c for c in cfgs if klass(c) == "last"]
facs = {"D = |pS - pT|": lambda c: abs(c[1] - c[2]),
        "signed pS - pT": lambda c: c[1] - c[2],
        "pM": lambda c: c[0],
        "pS": lambda c: c[1],
        "pT": lambda c: c[2],
        "pM - pS (g + 1)": lambda c: c[0] - c[1],
        "pM - pT": lambda c: c[0] - c[2],
        "g = 0 or not": lambda c: c[0] - c[1] == 1,
        "(pM - pS, pM - pT) [ages at the record]": lambda c: (c[0] - c[1], c[0] - c[2]),
        "(D, g = 0 or not)": lambda c: (abs(c[1] - c[2]), c[0] - c[1] == 1),
        "(signed sep, pM = 9 or not)": lambda c: (c[1] - c[2], c[0] == 9)}
for b in range(4):
    e = np.array([errT(c, b) for c in late])
    line = []
    for name, f in facs.items():
        keys = [f(c) for c in late]
        fit = np.zeros(len(late))
        for kk in set(keys):
            m = np.array([k_ == kk for k_ in keys])
            fit[m] = e[m].mean()
        line.append(f"{name}: {1 - ((e - fit) ** 2).sum() / ((e - e.mean()) ** 2).sum():.2f} ({len(set(keys))} groups)")
    P(f"  bin {b} (mean rate {e.mean():.3f}, sd over configs {e.std():.3f}): " + "; ".join(line))

for pMfix in (9, 8, 7, 6):
    for b in (1, 2):
        P(f"\n[4] pM = {pMfix} fixed: P(answer = pT) in bin {['(0,.03]', '(.03,.08]', '(.08,.2]', '(.2,.5]'][b]}; rows pS (runner-up), columns pT (third)")
        P("        pT: " + " ".join(f"{q:5d}" for q in range(pMfix)))
        for pS in range(pMfix):
            P(f"    pS={pS}:  " + " ".join("  -  " if pS == pT else f"{errT((pMfix, pS, pT), b):.3f}" for pT in range(pMfix)))

P("\n[5] max FIRST, pM = 0 fixed: P(answer = pT) in bin (.03,.08]; rows pS, columns pT")
P("        pT: " + " ".join(f"{q:5d}" for q in range(1, 10)))
for pS in range(1, 10):
    P(f"    pS={pS}:  " + " ".join("  -  " if pS == pT else f"{errT((0, pS, pT), 1):.3f}" for pT in range(1, 10)))
P("    same, bin (.08,.2]")
for pS in range(1, 10):
    P(f"    pS={pS}:  " + " ".join("  -  " if pS == pT else f"{errT((0, pS, pT), 2):.3f}" for pT in range(1, 10)))

P("\n[6] same positions of the two candidates (0 and 7, 0 and 8, 1 and 8), every position of the max: P(answer = pT) per bin")
for a, b_ in ((0, 7), (0, 8), (1, 8), (1, 7), (2, 8), (0, 6), (3, 4), (3, 6)):
    for pS, pT in ((a, b_), (b_, a)):
        for pM in range(10):
            if pM in (pS, pT):
                continue
            c = (pM, pS, pT)
            P(f"    (pS,pT)=({pS},{pT}) pM={pM} [{klass(c):5s}]: " + " ".join(f"{errT(c, b):.3f}" for b in range(4)))

P("\n[7] outcome distribution in the worst max-last configurations, bin (.08,.2] and (.2,.5]: P(pS) P(pT) P(pM) P(4th) P(other)")
e2 = np.array([errT(c, 2) for c in late])
for i in np.argsort(-e2)[:12]:
    c = late[i]
    P(f"    (pS,pT,pM)=({c[1]},{c[2]},{c[0]}) D={abs(c[1]-c[2])}: "
      + " | ".join(" ".join(f"{v:.3f}" for v in cnt[c[0], c[1], c[2], b] / nbin[b]) for b in (2, 3)))
P("    pooled over the configurations of each D (max last), bin (.08,.2]: P(pS) P(pT) P(pM) P(4th) P(other); share of all errors that are 'answers the third'")
for d in range(1, 9):
    cs = [c for c in late if abs(c[1] - c[2]) == d]
    for b in (0, 1, 2, 3):
        o = np.sum([cnt[c[0], c[1], c[2], b] for c in cs], 0) / (len(cs) * nbin[b])
        P(f"    D={d} bin {b}: " + " ".join(f"{v:.3f}" for v in o) + f"   share of errors that are pT: {o[1] / (1 - o[0]):.2f}")

P("\n[8] share of the D4 'answers the third' errors (max last) carried by configurations with the runner-up right before the max (pS = pM - 1, g = 0) and by pM = 9")
for d in range(1, 9):
    cs = [c for c in late if abs(c[1] - c[2]) == d]
    line = []
    for b in range(4):
        tot = sum(cnt[c[0], c[1], c[2], b, 1] for c in cs)
        g0 = sum(cnt[c[0], c[1], c[2], b, 1] for c in cs if c[0] - c[1] == 1)
        m9 = sum(cnt[c[0], c[1], c[2], b, 1] for c in cs if c[0] == 9)
        line.append(f"{g0 / max(tot, 1):.2f}/{m9 / max(tot, 1):.2f} (n_err {tot})")
    ng0 = sum(c[0] - c[1] == 1 for c in cs); n9 = sum(c[0] == 9 for c in cs)
    P(f"    D={d}: g=0 share / pM=9 share per bin: " + "  ".join(line) + f"   [configs: {len(cs)}, of which g=0: {ng0}, pM=9: {n9}]")
P("    D4 with the g = 0 configurations removed (equal weight per configuration):")
for d in range(1, 9):
    cs = [c for c in late if abs(c[1] - c[2]) == d and c[0] - c[1] >= 2]
    P(f"    D={d} ({len(cs):3d} configs): " + " ".join(f"{np.mean([errT(c, b) for c in cs]):.3f}" for b in range(4)))
P("    D4 with g = 0 AND pM = 9 removed:")
for d in range(1, 9):
    cs = [c for c in late if abs(c[1] - c[2]) == d and c[0] - c[1] >= 2 and c[0] < 9]
    if cs:
        P(f"    D={d} ({len(cs):3d} configs): " + " ".join(f"{np.mean([errT(c, b) for c in cs]):.3f}" for b in range(4)))
P("    D4 for pM = 9 only, g >= 1:")
for d in range(1, 9):
    cs = [c for c in late if abs(c[1] - c[2]) == d and c[0] - c[1] >= 2 and c[0] == 9]
    if cs:
        P(f"    D={d} ({len(cs):3d} configs): " + " ".join(f"{np.mean([errT(c, b) for c in cs]):.3f}" for b in range(4)))

P("\n[9] record-time failure vs final answer (max last): per D, per bin: P(y_rec < 0) | P(final = pT) | P(final = pT and y_rec < 0) | P(final = pT and y_rec > 0)")
for d in range(1, 9):
    cs = [c for c in late if abs(c[1] - c[2]) == d]
    line = []
    for b in range(4):
        r = np.sum([rec[c[0], c[1], c[2], b] for c in cs], 0) / (len(cs) * nbin[b])
        line.append(f"{r[0]:.3f} | {r[1] + r[2]:.3f} | {r[1]:.3f} | {r[2]:.3f}")
    P(f"    D={d}: " + "   ||   ".join(line))
out.close()
