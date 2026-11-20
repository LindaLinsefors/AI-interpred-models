"""Verifier V1: claim T4 (score laws).  States with S_t > 0.2, t = 2..9, units of M_t:
    d = (s[p] - s[p2]) / M_t,   v = (x_p - S_t) / M_t,   s = U h_t,  index of position p is 9 - t + p.
Seeds 5201..5205, 200k sequences each (1M sequences).  Run: python -B v1_t4.py -> v1_t4.txt
With --save it also writes the pooled (d, v, age, kind, t, a2) table to v1_t4_table.npz (0.5 GB), which
v1_t4_extra.py reads; the table was deleted after use.
"""
from vlib import *

P = Tee("v1_t4.txt")
SEEDS = [5201, 5202, 5203, 5204, 5205]
NB = 200_000
D, V, AGE, KIND, TT, A2, SEED = [], [], [], [], [], [], []
for seed in SEEDS:
    x = sample(NB, seed)
    H = run(x)
    for t in range(2, 10):
        sub, _ = scores(H, t)                       # (B, t+1), column = position
        p, val = top3(x, t)
        p1, p2 = p[:, 0], p[:, 1]
        M, S = val[:, 0], val[:, 1]
        keep = S > 0.2
        sub, xx, p1, p2, M, S = sub[keep], x[keep, : t + 1], p1[keep], p2[keep], M[keep], S[keep]
        r = np.arange(len(M))
        d = (sub - sub[r, p2][:, None]) / M[:, None]
        v = (xx - S[:, None]) / M[:, None]
        pos = np.broadcast_to(np.arange(t + 1), d.shape)
        kind = np.zeros(d.shape, np.int8)           # 0 other, 1 max, 2 runner-up
        kind[r, p1] = 1
        kind[r, p2] = 2
        sel = kind != 2
        D.append(d[sel].astype(np.float32)); V.append(v[sel].astype(np.float32))
        AGE.append((t - pos)[sel].astype(np.int8)); KIND.append(kind[sel])
        TT.append(np.full(sel.sum(), t, np.int8))
        A2.append(np.broadcast_to((t - p2)[:, None], d.shape)[sel].astype(np.int8))
        SEED.append(np.full(sel.sum(), seed, np.int16))
    del H
D, V, AGE, KIND, TT, A2, SEED = map(np.concatenate, (D, V, AGE, KIND, TT, A2, SEED))
if "--save" in sys.argv:        # 0.5 GB cache for v1_t4_extra.py; not kept in the folder by default
    np.savez(HERE / "v1_t4_table.npz", d=D, v=V, age=AGE, kind=KIND, t=TT, a2=A2, seed=SEED)
P("entries:", len(D), " seeds", SEEDS, f"{NB} sequences each")


def binned(mask, edges, label, seeds=True):
    """median d, median v, ratio, n per bin (edges[k], edges[k+1]]; plus across-seed sd of the median d."""
    d, v, sd = D[mask], V[mask], SEED[mask]
    b = np.searchsorted(edges, v, side="left") - 1
    P(f"--- {label}")
    P("   bin (lo, hi]        n      med v     med d    med d/med v   med(d/v)   P(d<0)   seed-sd(med d)")
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() < 50:
            continue
        md, mv = np.median(d[m]), np.median(v[m])
        ss = np.std([np.median(d[m & (sd == s)]) for s in SEEDS], ddof=1) if seeds else np.nan
        P(f"   ({edges[k]:6.2f},{edges[k+1]:6.2f}] {m.sum():8d} {mv:9.4f} {md:9.3f} {md/mv:10.2f} {np.median(d[m]/v[m]):10.2f} {np.mean(d[m]<0):8.4f} {ss:8.3f}")
        rows.append((edges[k], edges[k + 1], m.sum(), mv, md))
    return rows


neg_edges = np.array([-np.inf, -2, -1.5, -1, -0.7, -0.5, -0.35, -0.25, -0.15, -0.10, -0.05, -0.02, 0])
max_edges = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.85, 1.0])

P("\n=== T4a: neither max nor runner-up, AGE 0 ===  claim: median d = 61.6 v on [-2,-0.05], per-bin ratio 55.9..61.8")
m0 = (KIND == 0) & (AGE == 0)
binned(m0, neg_edges, "other, age 0")
rng_m = m0 & (V >= -2) & (V <= -0.05)
d, v = D[rng_m].astype(np.float64), V[rng_m].astype(np.float64)
P(f"   on v in [-2,-0.05]: n={len(d)}  LS slope through origin = {(d*v).sum()/(v*v).sum():.2f};  LS with intercept: slope {np.polyfit(v,d,1)[0]:.2f} intercept {np.polyfit(v,d,1)[1]:.2f};"
  f"  median(d/v) = {np.median(d/v):.2f};  ratio of medians = {np.median(d)/np.median(v):.2f};  centred R2 of d=61.6v: {1-((d-61.6*v)**2).sum()/((d-d.mean())**2).sum():.4f}")
q = np.percentile(d / v, [5, 25, 50, 75, 95])
P("   percentiles of d/v (5,25,50,75,95):", np.round(q, 1))
for s in SEEDS:
    mm = rng_m & (SEED == s)
    dd, vv = D[mm].astype(np.float64), V[mm].astype(np.float64)
    P(f"     seed {s}: LS0 slope {(dd*vv).sum()/(vv*vv).sum():.2f}  median(d/v) {np.median(dd/vv):.2f}")
# finer bins to bracket the per-bin ratio statement
fine = np.array([-2, -1.75, -1.5, -1.25, -1, -0.85, -0.7, -0.6, -0.5, -0.42, -0.35, -0.3, -0.25, -0.2, -0.15, -0.125, -0.10, -0.075, -0.05])
binned(m0, fine, "other, age 0, finer bins", seeds=False)
P("   age 0 by t (LS0 slope on [-2,-0.05], median d/v):")
for t in range(2, 10):
    mm = rng_m & (TT == t)
    dd, vv = D[mm].astype(np.float64), V[mm].astype(np.float64)
    P(f"     t={t}: n={len(dd)} LS0 {(dd*vv).sum()/(vv*vv).sum():.2f}  med(d/v) {np.median(dd/vv):.2f}")
P("   age 0 by age of the runner-up a2 (LS0 slope on [-2,-0.05], median d/v):")
for a in range(1, 10):
    mm = rng_m & (A2 == a)
    dd, vv = D[mm].astype(np.float64), V[mm].astype(np.float64)
    P(f"     a2={a}: n={len(dd)} LS0 {(dd*vv).sum()/(vv*vv).sum():.2f}  med(d/v) {np.median(dd/vv):.2f}")

P("\n=== T4b: other, AGE 1 ===  claim: -4.7 (-0.10,-0.05]; -9.7 (-0.25,-0.15]; -12.8 (-0.7,-0.5]; -18.3 v<-2")
binned((KIND == 0) & (AGE == 1), neg_edges, "other, age 1")
P("\n=== T4c: other, AGES >= 2 ===  claim: -1.25 (-0.02,0]; -5.3 (-0.10,-0.05]; -10.8 (-0.25,-0.15]; -17.7 (-0.7,-0.5]; -23.7 v<-2")
binned((KIND == 0) & (AGE >= 2), neg_edges, "other, ages >= 2")
P("   by single age, median d in bins (-0.10,-0.05] | (-0.25,-0.15] | (-0.7,-0.5] | v<-2 :")
for a in range(0, 10):
    mm = (KIND == 0) & (AGE == a)
    row = []
    for lo, hi in ((-0.10, -0.05), (-0.25, -0.15), (-0.7, -0.5), (-np.inf, -2)):
        k = mm & (V > lo) & (V <= hi)
        row.append(f"{np.median(D[k]):8.2f}" if k.sum() > 50 else "     nan")
    P(f"     age {a}: " + " | ".join(row))

P("\n=== T4d: the max (p = p1), v = (M-S)/M ===  claim: age 0: -9.5 (0.1,0.2]; -25.0 (0.35,0.5]; -48.8 (0.85,1]; age 1: -26.4 (0.85,1]; ages 2..9, v>0.2: medians in -4.5..-14.5")
binned((KIND == 1) & (AGE == 0), max_edges, "max, age 0")
binned((KIND == 1) & (AGE == 1), max_edges, "max, age 1")
binned((KIND == 1) & (AGE >= 2), max_edges, "max, ages >= 2 pooled")
P("   max, single ages 2..9: median d per bin of v " + " ".join(f"({max_edges[k]:.2f},{max_edges[k+1]:.2f}]" for k in range(len(max_edges) - 1)))
lo_all, hi_all = np.inf, -np.inf
for a in range(2, 10):
    mm = (KIND == 1) & (AGE == a)
    b = np.searchsorted(max_edges, V[mm], side="left") - 1
    row = []
    for k in range(len(max_edges) - 1):
        k_ = b == k
        if k_.sum() > 50:
            md = np.median(D[mm][k_]); row.append(f"{md:8.2f}")
            if max_edges[k] >= 0.2:
                lo_all, hi_all = min(lo_all, md), max(hi_all, md)
        else:
            row.append("     nan")
    allv = np.median(D[mm][V[mm] > 0.2])
    P(f"     age {a}: " + " ".join(row) + f"   | all v>0.2: {allv:.2f}  n={int((V[mm] > 0.2).sum())}")
P(f"   range of per-(age, bin) medians for ages 2..9 and v > 0.2: {lo_all:.2f} .. {hi_all:.2f}")
P("   per-age medians over all v>0.2 listed in the last column above")
