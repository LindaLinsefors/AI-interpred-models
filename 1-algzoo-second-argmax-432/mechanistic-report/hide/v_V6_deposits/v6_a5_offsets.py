"""V6 attack: does the cosine-law lead predict where the network answers the third?

Per max-last configuration (a2_configs.npz): c1, c2 of the D1 regression at the record step, the
break-even relative gap r* = -c2 / (c1 + c2)  [y = 0 at (M0 - S0)/M0 = -c2/c1; in D4 units (S-T)/S],
P(y < 0) at the record and the final P(answer = pT).  Rank correlations over configurations.
Also: the cosine law fitted separately for 'old runner-up older / younger than the old max' and for
'old runner-up arrived on the step before the record' (aS = 1).
"""
from v6_data import *
from scipy.stats import spearmanr

out = open(HERE / "a5_offsets.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


z = np.load(HERE / "a2_configs.npz")
reg, cnt, nbin, rec = z["reg"], z["cnt"], z["nbin"], z["rec"]
late = [(t, pm, ps) for t in range(2, 10) for pm in range(t) for ps in range(t) if pm != ps]   # (t = pM, pM0 = pS, pS0 = pT)
c1 = np.array([reg[c][0] for c in late]); c2 = np.array([reg[c][1] for c in late]); py = np.array([reg[c][3] for c in late])
aM = np.array([c[0] - c[1] for c in late]); aS = np.array([c[0] - c[2] for c in late]); tt = np.array([c[0] for c in late])
Dl = np.abs(aM - aS)
errT = np.array([[cnt[c[0], c[1], c[2], b, 1] / nbin[b] for b in range(4)] for c in late])
yneg = np.array([[rec[c[0], c[1], c[2], b, 0] / nbin[b] for b in range(4)] for c in late])
law = 45 * (1 - np.cos(OMEGA * Dl))

P("[1] c2 by (aM, aS), mean over t  [rows aM = age of old max at the record, columns aS = age of old runner-up]")
P("       aS: " + " ".join(f"{a:6d}" for a in range(1, 10)))
for a in range(1, 10):
    P(f"    aM={a}: " + " ".join(f"{c2[(aM == a) & (aS == b)].mean():6.2f}" if ((aM == a) & (aS == b)).any() else "     -" for b in range(1, 10)))
P("    break-even relative gap -c2/c1 (the old runner-up is ahead on average when (M0 - S0)/M0 is below this; negative = never)")
for a in range(1, 10):
    P(f"    aM={a}: " + " ".join(f"{(-c2 / c1)[(aM == a) & (aS == b)].mean():6.3f}" if ((aM == a) & (aS == b)).any() else "     -" for b in range(1, 10)))
P("    final P(answer = pT), bin (.03,.08], by (aM, aS) at the record, mean over t (i.e. over pM)")
for a in range(1, 10):
    P(f"    aM={a}: " + " ".join(f"{errT[(aM == a) & (aS == b), 1].mean():6.3f}" if ((aM == a) & (aS == b)).any() else "     -" for b in range(1, 10)))
P("    same, only pM = 9 (record on the last step; the answer is read at once)")
for a in range(1, 10):
    P(f"    aM={a}: " + " ".join(f"{errT[(aM == a) & (aS == b) & (tt == 9), 1].mean():6.3f}" if ((aM == a) & (aS == b) & (tt == 9)).any() else "     -" for b in range(1, 10)))
P("    P(y < 0) at the record step, bin (.03,.08], by (aM, aS), mean over t")
for a in range(1, 10):
    P(f"    aM={a}: " + " ".join(f"{yneg[(aM == a) & (aS == b), 1].mean():6.3f}" if ((aM == a) & (aS == b)).any() else "     -" for b in range(1, 10)))
P("    spread over t of that P(y < 0) (max - min)")
for a in range(1, 10):
    P(f"    aM={a}: " + " ".join(f"{np.ptp(yneg[(aM == a) & (aS == b), 1]):6.3f}" if ((aM == a) & (aS == b)).sum() > 1 else "     -" for b in range(1, 10)))

P("\n[2] Spearman rank correlation over configurations between a failure rate and candidate predictors")
for nm, sel in (("all 240 max-last configurations", np.ones(len(late), bool)), ("g >= 1 (204)", aM >= 2),
                ("g >= 1 and aS >= 2 (168)", (aM >= 2) & (aS >= 2)), ("pM = 9, g >= 1 (56)", (tt == 9) & (aM >= 2))):
    P(f"    {nm}:")
    for tn, target in (("P(y<0) at record, bin (.03,.08]", yneg[:, 1]), ("P(y<0) at record, bin (.08,.2]", yneg[:, 2]),
                       ("final P(ans = pT), bin (.03,.08]", errT[:, 1]), ("final P(ans = pT), bin (.08,.2]", errT[:, 2])):
        r = [spearmanr(target[sel], v[sel])[0] for v in (-law, -c1, -c2 / c1, Dl)]
        P(f"       {tn:34s}: with -(cosine-law lead) {r[0]:+.2f} | with -(actual c1) {r[1]:+.2f} | with break-even gap -c2/c1 {r[2]:+.2f} | with Delta {r[3]:+.2f}")

P("\n[3] cosine law K (1 - cos(42.75 Delta)) fitted to the per-configuration c1 in sub-families (R2 over configurations; 'const' = R2 of the mean is 0 by definition)")
fam = {"old runner-up OLDER than old max (aS > aM), aM >= 2": (aS > aM) & (aM >= 2),
       "old runner-up YOUNGER (aS < aM), aS >= 2": (aS < aM) & (aS >= 2),
       "old runner-up arrived the step before the record (aS = 1), aM >= 2": (aS == 1) & (aM >= 2),
       "g = 0 (aM = 1; excluded by D1)": aM == 1,
       "both ages >= 2": (aS >= 2) & (aM >= 2),
       "all g >= 1": aM >= 2}
for nm, m in fam.items():
    f = 1 - np.cos(OMEGA * Dl[m]); y = c1[m]
    K = f @ y / (f @ f)
    r2 = 1 - ((y - K * f) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    X = np.column_stack([f, np.ones(m.sum())]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2b = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    P(f"    {nm:68s} n = {m.sum():3d}: K = {K:5.1f}, R2 = {r2:6.3f} | with intercept: {b[0]:5.1f} (1-cos) + {b[1]:5.1f}, R2 = {r2b:.3f} | mean c1 {y.mean():5.1f}, range {y.min():.1f}..{y.max():.1f}")
P("    c1 in the aS = 1 column vs the cosine law (K = 45):")
for a in range(2, 10):
    m = (aM == a) & (aS == 1)
    P(f"       aM = {a} (Delta = {a - 1}): c1 = {c1[m].mean():5.1f}   law {45 * (1 - np.cos(OMEGA * (a - 1))):5.1f}   c2 = {c2[m].mean():5.2f}   P(y>0) = {py[m].mean():.3f}")
out.close()
