"""V6 attack on the regression design: a regression-free decomposition with exact input traces.

The net is piecewise linear and bias-free, so with the gates of a given input held fixed
    s_9[i] = sum_p G[i, p] x_p,   G[i, p] = (U J[9, p])[i]      (common.traces).
For a record at t = 9 (old max M0 at pM0, old runner-up S0 at pS0, new max x_9):
    y = s_9[pM0] - s_9[pS0]
      = (G[pM0,pM0] - G[pS0,pM0]) M0 + (G[pM0,pS0] - G[pS0,pS0]) S0 + (G[pM0,9] - G[pS0,9]) x_9 + rest
so the regression's c1 + c2 should equal E[G[pM0,pM0] - G[pS0,pM0]] and c1 should equal
E[G[pS0,pS0] - G[pM0,pS0]] if the two-regressor design is not distorted by what it omits.
Also stored: the mean footprint G[., pM0] and G[., pS0] over all ten slots, to test 'first harmonic in position'.

Controlled sampling as in v6_a2_configs.py (t = 9 only: 72 configurations, 20k base sequences, seed 5300).
"""
from v6_data import *

out = open(HERE / "a8_traces.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


SEED, B = 5300, 20_000
rng = np.random.default_rng(SEED)
v = -np.sort(-rng.standard_normal((B, 10)), axis=1)
rest = rng.permuted(v[:, 3:], axis=1)
Mv, Sv, Tv = v[:, 0], v[:, 1], v[:, 2]              # new max, old max (M0), old runner-up (S0)
sel = (Sv > 0.3) & (Tv > 0.1)
ar = np.arange(B)
res = {}
for pm in range(9):
    for ps in range(9):
        if pm == ps:
            continue
        free = [p for p in range(9) if p not in (pm, ps)]
        x = np.empty((B, 10))
        x[:, 9], x[:, pm], x[:, ps] = Mv, Sv, Tv
        x[:, free] = rest
        H, J = traces(x)
        G = np.einsum("iu,bpu->bip", U, J[:, 9])     # (B, slot i, input p)
        s = H[:, 9] @ U.T
        y = s[:, pm] - s[:, ps]
        aMM, aSM = G[:, pm, pm], G[:, ps, pm]        # footprint of the old max at its own slot / at the runner-up's slot
        aSS, aMS = G[:, ps, ps], G[:, pm, ps]        # footprint of the old runner-up at its own slot / at the old max's slot
        aX = G[:, pm, 9] - G[:, ps, 9]
        other = y - (aMM - aSM) * Sv - (aMS - aSS) * Tv - aX * Mv
        b, r2c, _ = ols([(Sv - Tv)[sel], Sv[sel]], y[sel])
        res[(pm, ps)] = dict(c1=b[0], c2=b[1], r2=r2c,
                             tM=np.mean((aMM - aSM)[sel]), tS=np.mean((aSS - aMS)[sel]),
                             sdM=np.std((aMM - aSM)[sel]), sdS=np.std((aSS - aMS)[sel]),
                             aSS=np.mean(aSS[sel]), aMS=np.mean(aMS[sel]), aMM=np.mean(aMM[sel]), aSM=np.mean(aSM[sel]),
                             aX=np.mean(aX[sel]), xterm=np.mean((aX * Mv)[sel] / Sv[sel]), other=np.mean(other[sel] / Sv[sel]),
                             sd_other=np.std(other[sel] / Sv[sel]), sd_x=np.std((aX * Mv)[sel] / Sv[sel]),
                             fM=G[sel][:, :, pm].mean(0), fS=G[sel][:, :, ps].mean(0), fX=G[sel][:, :, 9].mean(0),
                             comp_on=(H[sel, 9][:, COMP] > 0).mean())
    print("pM0", pm, "done", flush=True)

P(f"records at t = 9, seed {SEED}, {B} base sequences per configuration, {sel.sum()} pass the D1 conditions")
P("ages at the record: aM = 9 - pM0, aS = 9 - pS0")
P("\n[1] regression c1 vs trace coefficient E[G[pS0,pS0] - G[pM0,pS0]]  (rows aM, columns aS; each cell 'regression / trace')")
P("        aS: " + " ".join(f"{a:13d}" for a in range(1, 10)))
for a in range(1, 10):
    row = []
    for b_ in range(1, 10):
        k = (9 - a, 9 - b_)
        row.append(f"{res[k]['c1']:6.1f}/{res[k]['tS']:6.1f}" if k in res else "            -")
    P(f"    aM={a}: " + " ".join(row))
c1 = np.array([r["c1"] for r in res.values()]); tS = np.array([r["tS"] for r in res.values()])
c12 = np.array([r["c1"] + r["c2"] for r in res.values()]); tM = np.array([r["tM"] for r in res.values()])
P(f"    over the 72 configurations: corr(c1, trace) = {np.corrcoef(c1, tS)[0, 1]:.4f}, max |c1 - trace| = {np.abs(c1 - tS).max():.1f}, "
  f"median |diff| = {np.median(np.abs(c1 - tS)):.2f}")
P(f"    c1 + c2 vs E[G[pM0,pM0] - G[pS0,pM0]]: corr = {np.corrcoef(c12, tM)[0, 1]:.4f}, max |diff| = {np.abs(c12 - tM).max():.1f}, "
  f"median |diff| = {np.median(np.abs(c12 - tM)):.2f}")

P("\n[2] where does the runner-up's coefficient come from? own-slot footprint E[G[pS0,pS0]] and cross footprint E[G[pM0,pS0]] (rows aM, columns aS; 'own / cross')")
for a in range(1, 10):
    row = []
    for b_ in range(1, 10):
        k = (9 - a, 9 - b_)
        row.append(f"{res[k]['aSS']:6.1f}/{res[k]['aMS']:6.1f}" if k in res else "            -")
    P(f"    aM={a}: " + " ".join(row))
P("    same for the old max: own-slot E[G[pM0,pM0]] / cross E[G[pS0,pM0]]")
for a in range(1, 10):
    row = []
    for b_ in range(1, 10):
        k = (9 - a, 9 - b_)
        row.append(f"{res[k]['aMM']:6.1f}/{res[k]['aSM']:6.1f}" if k in res else "            -")
    P(f"    aM={a}: " + " ".join(row))

P("\n[3] the terms the two-regressor design leaves out, in units of M0: mean (sd) of the new max's term (G[pM0,9]-G[pS0,9]) x_9 / M0, and of the other seven inputs' term")
P("    rows aM, columns aS: 'x9-term mean(sd) | others mean(sd)'")
for a in range(1, 10):
    row = []
    for b_ in range(1, 10):
        k = (9 - a, 9 - b_)
        row.append(f"{res[k]['xterm']:5.1f}({res[k]['sd_x']:4.1f})|{res[k]['other']:5.1f}({res[k]['sd_other']:4.1f})" if k in res else "                       -")
    P(f"    aM={a}: " + " ".join(row))

P("\n[4] 'first harmonic in position': mean footprint of the old max over the slots of the older inputs, G[i, pM0], i = 0..8,")
P("    fitted by a cos(42.75 (i - pM0) - phi) + b (least squares over the slots i != pM0 ... all nine slots used). Shown for aS-averaged footprints per aM.")
ii = np.arange(9)
for a in range(1, 10):
    pm = 9 - a
    fs = [res[(pm, ps)]["fM"][:9] for ps in range(9) if ps != pm]
    f = np.mean(fs, 0)
    X = np.column_stack([np.cos(OMEGA * (ii - pm)), np.sin(OMEGA * (ii - pm)), np.ones(9)])
    b = np.linalg.lstsq(X, f, rcond=None)[0]
    r2 = 1 - ((f - X @ b) ** 2).sum() / ((f - f.mean()) ** 2).sum()
    # same without the own slot (the own slot also carries the input's own 'tape' entry)
    m = ii != pm
    b2 = np.linalg.lstsq(X[m], f[m], rcond=None)[0]
    r22 = 1 - ((f[m] - X[m] @ b2) ** 2).sum() / ((f[m] - f[m].mean()) ** 2).sum()
    P(f"    aM={a} (slot {pm}): footprint over slots 0..8 = " + " ".join(f"{v_:6.1f}" for v_ in f)
      + f" | all 9 slots: amp {np.hypot(b[0], b[1]):5.1f}, phase {np.degrees(np.arctan2(b[1], b[0])):6.1f} deg, R2 {r2:.3f}"
      + f" | without own slot: amp {np.hypot(b2[0], b2[1]):5.1f}, phase {np.degrees(np.arctan2(b2[1], b2[0])):6.1f}, R2 {r22:.3f}")
P("    footprint of the old runner-up, G[i, pS0], averaged over aM, per aS:")
for a in range(1, 10):
    ps = 9 - a
    fs = [res[(pm, ps)]["fS"][:9] for pm in range(9) if pm != ps]
    f = np.mean(fs, 0)
    X = np.column_stack([np.cos(OMEGA * (ii - ps)), np.sin(OMEGA * (ii - ps)), np.ones(9)])
    b = np.linalg.lstsq(X, f, rcond=None)[0]
    r2 = 1 - ((f - X @ b) ** 2).sum() / ((f - f.mean()) ** 2).sum()
    P(f"    aS={a} (slot {ps}): footprint over slots 0..8 = " + " ".join(f"{v_:6.1f}" for v_ in f)
      + f" | amp {np.hypot(b[0], b[1]):5.1f}, phase {np.degrees(np.arctan2(b[1], b[0])):6.1f} deg, R2 {r2:.3f}")
P("    mean fraction of comparators ON at the record step (D1 records, t = 9): " + f"{np.mean([r['comp_on'] for r in res.values()]):.4f}")
np.savez_compressed(HERE / "a8_traces.npz", keys=np.array(list(res.keys())),
                    fM=np.array([r["fM"] for r in res.values()]), fS=np.array([r["fS"] for r in res.values()]),
                    fX=np.array([r["fX"] for r in res.values()]), c1=c1, tS=tS, c12=c12, tM=tM)
out.close()
