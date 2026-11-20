"""R4 / r03: section 5.4 (new maximum): gate sequence, amplitudes of units 5 and 9, thresholds one
step later, the old maximum's own unit-5 pulse, unit 9 / unit 0 interventions.
Seeds 7041-7046.  Output: r03_newmax_out.txt
"""
from rlib import *

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


def mask_hook(units, mask, mode):
    """mask (B, T) bool: at those (row, step) set unit to 0 ('zero') or to its pre-activation ('linear')."""
    def hook(t, z, h):
        m = mask[:, t]
        if m.any():
            for u in units:
                h[m, u] = 0.0 if mode == "zero" else z[m, u]
        return h
    return hook


def step_mask(B, step):
    """(B, 10) mask that is True at column step[b] (rows with step outside 0..9 get nothing)."""
    m = np.zeros((B, 10), bool)
    ok = (step >= 0) & (step <= 9)
    m[np.nonzero(ok)[0], step[ok]] = True
    return m


N = 400_000
# ============================================================ part 1: anatomy of a new maximum
cnt = {}


def add(k, v):
    cnt[k] = cnt.get(k, 0) + v


rows_reg = {k: [] for k in ("X", "Xraw", "h5", "h9", "L1", "L2", "g", "th", "Xth", "h4", "z0")}
gen = np.zeros((10, 8))      # by s (any s=1..8, M0>0): n, u4 on, all comps off, u5 on s+1, u9 on s+1, n(s<=7), u0 off s+2, spare
first = np.zeros(4)          # first positive input (no positive before): n, u4 on, all comps off
for sd in (7041, 7042, 7043):
    x = gauss(N, sd)
    H, Z = fwd(x)
    st = prefix_stats(x)
    Mp, Sp, pMp = prev(st["M"], -np.inf), prev(st["S"], -np.inf), prev(st["pM"], -1)
    for s in range(0, 9):
        new = x[:, s] > Mp[:, s]
        f = new & (x[:, s] > 0) & (Mp[:, s] <= 0)
        first += [f.sum(), (Z[f, s, 4] > 0).sum(), (Z[f, s][:, COMP] <= 0).all(1).sum(), 0]
        if s == 0:
            continue
        rec = new & (Mp[:, s] > 0)
        gen[s, 0] += rec.sum(); gen[s, 1] += (Z[rec, s, 4] > 0).sum()
        gen[s, 2] += (Z[rec, s][:, COMP] <= 0).all(1).sum()
        gen[s, 3] += (Z[rec, s + 1, 5] > 0).sum(); gen[s, 4] += (Z[rec, s + 1, 9] > 0).sum()
        if s <= 7:
            gen[s, 5] += rec.sum(); gen[s, 6] += (Z[rec, s + 2, 0] <= 0).sum()
    for s in range(2, 7):
        rec = (x[:, s] > Mp[:, s]) & (Mp[:, s] > 0)
        Mn, M0, S0 = x[rec, s], Mp[rec, s], Sp[rec, s]
        S0p = np.maximum(S0, 0)
        nxt = x[rec, s + 1:s + 4]
        L1 = (nxt < M0[:, None]).all(1)
        L2 = (nxt < S0p[:, None]).all(1)
        z = Z[rec]
        add("n", rec.sum()); add("L1", L1.sum()); add("L2", L2.sum())
        add("u4_s", (z[:, s, 4] > 0).sum())
        offc = z[:, s][:, COMP] <= 0
        add("comp_all_off_s", offc.all(1).sum())
        for i, j in enumerate(COMP):
            add(f"comp{j}_on_s", (~offc[:, i]).sum())
        add("u5_s1", (z[:, s + 1, 5] > 0).sum()); add("u9_s1", (z[:, s + 1, 9] > 0).sum())
        add("u5_s1_L1", (z[L1, s + 1, 5] > 0).sum()); add("u9_s1_L1", (z[L1, s + 1, 9] > 0).sum())
        add("u0off_s2", (z[:, s + 2, 0] <= 0).sum()); add("u0off_s2_L1", (z[L1, s + 2, 0] <= 0).sum())
        add("u0off_s2_L2", (z[L2, s + 2, 0] <= 0).sum())
        add("u0on_s1_L1", (z[L1, s + 1, 0] > 0).sum()); add("u0on_s3_L1", (z[L1, s + 3, 0] > 0).sum())
        add("comps_on_s1_L1", (z[L1, s + 1][:, COMP] > 0).all(1).sum())
        add("u5_s2_L1", (z[L1, s + 2, 5] > 0).sum()); add("u9_s2_L1", (z[L1, s + 2, 9] > 0).sum())
        rows_reg["X"].append(np.c_[Mn - M0, M0, S0p]); rows_reg["Xraw"].append(np.c_[Mn - M0, M0, S0])
        rows_reg["h5"].append(H[rec, s + 1, 5]); rows_reg["h9"].append(H[rec, s + 1, 9]); rows_reg["h4"].append(H[rec, s, 4])
        rows_reg["L1"].append(L1); rows_reg["L2"].append(L2); rows_reg["g"].append(s - pMp[rec, s])
        rows_reg["z0"].append(Z[rec, s + 2, 0])
        pre = H[rec, s] @ W.T                                    # W h_s -> thresholds at s+1
        rows_reg["th"].append(np.c_[pre[:, COMP] / G, -pre[:, 4] / W4])
        rows_reg["Xth"].append(np.c_[Mn, M0, S0p])
R = {k: np.concatenate(v) for k, v in rows_reg.items()}
n = cnt["n"]
P("=== SECTION 5.4, part 1: new maximum at s = 2..6 with old maximum M0 > 0 (n = %d, seeds 7041-7043)" % n)
P("L1 (next three inputs < M0): %.3f of records ; L2 (< max(S0,0)): %.3f" % (cnt["L1"] / n, cnt["L2"] / n))
P("step s  : unit 4 on %.5f ; all four comparators off %.4f ; comparator on-rates 1,6,7,8: %s" % (
    cnt["u4_s"] / n, cnt["comp_all_off_s"] / n, np.round([cnt[f"comp{j}_on_s"] / n for j in COMP], 4)))
P("step s+1: unit 5 on %.5f, unit 9 on %.4f (all records) ; under L1: %.5f, %.4f ; comparators all on under L1 %.4f" % (
    cnt["u5_s1"] / n, cnt["u9_s1"] / n, cnt["u5_s1_L1"] / cnt["L1"], cnt["u9_s1_L1"] / cnt["L1"], cnt["comps_on_s1_L1"] / cnt["L1"]))
P("step s+2: unit 0 OFF: all records %.4f ; under L1 %.4f ; under L2 %.4f   (unit 0 on at s+1 / s+3 under L1: %.4f / %.4f)" % (
    cnt["u0off_s2"] / n, cnt["u0off_s2_L1"] / cnt["L1"], cnt["u0off_s2_L2"] / cnt["L2"], cnt["u0on_s1_L1"] / cnt["L1"], cnt["u0on_s3_L1"] / cnt["L1"]))
P("          units 5 / 9 on at s+2 under L1: %.4f / %.4f" % (cnt["u5_s2_L1"] / cnt["L1"], cnt["u9_s2_L1"] / cnt["L1"]))
P("by step s = 1..8 (any following inputs, M0 > 0): n, unit 4 on, all comps off, unit 5 on at s+1, unit 9 on at s+1, unit 0 off at s+2")
for s in range(1, 9):
    r = gen[s]
    P("   s=%d n=%7d  %.5f  %.4f  %.5f  %.4f  %s" % (s, r[0], r[1] / r[0], r[2] / r[0], r[3] / r[0], r[4] / r[0], ("%.4f" % (r[6] / r[5])) if r[5] else "-"))
r = gen.sum(0)
P("   all n=%7d  %.5f  %.4f  %.5f  %.4f  %.4f" % (r[0], r[1] / r[0], r[2] / r[0], r[3] / r[0], r[4] / r[0], r[6] / r[5]))
P("first positive input (no positive input before), s = 0..8: n = %d, unit 4 on %.5f, all comparators off %.4f" % (first[0], first[1] / first[0], first[2] / first[0]))
P("all positive new maxima (s = 0..8): unit 4 on %.5f" % ((first[1] + gen[:, 1].sum()) / (first[0] + gen[:, 0].sum())))

for name, sel in (("all records", np.ones(len(R["L1"]), bool)), ("L1", R["L1"])):
    for u in ("h5", "h9"):
        c, r2, _ = lsq(R["X"][sel], R[u][sel])
        c2, r22, _ = lsq(R["Xraw"][sel], R[u][sel])
        P("%s(s+1) ~ a (Mn-M0) + b M0 + c S0+  [%s]: %.2f, %.2f, %.2f  R2 %.3f   | with raw S0: %.2f, %.2f, %.2f  R2 %.3f" % (
            u, name, c[0], c[1], c[2], r2, c2[0], c2[1], c2[2], r22))
P("mean h5(s+1), h9(s+1) under L1: %.2f, %.2f ;  W[0,5]*h5 = %.2f, W[0,9]*h9 = %.2f  (W[0,5] = %.3f, W[0,9] = %.3f) ; mean z0(s+2) %.2f" % (
    R["h5"][R["L1"]].mean(), R["h9"][R["L1"]].mean(), W[0, 5] * R["h5"][R["L1"]].mean(), W[0, 9] * R["h9"][R["L1"]].mean(), W[0, 5], W[0, 9], R["z0"][R["L1"]].mean()))
P("h5(s+1) under L1 by age g of the old max (coefficients of Mn-M0, M0, S0+; R2):")
for g in range(1, 7):
    sel = R["L1"] & (R["g"] == g)
    c, r2, _ = lsq(R["X"][sel], R["h5"][sel])
    c9, r29, _ = lsq(R["X"][sel], R["h9"][sel])
    P("   g=%d n=%6d  h5: %.2f %.2f %.2f (R2 %.3f)   h9: %.2f %.2f %.2f (R2 %.3f)" % (g, sel.sum(), *c, r2, *c9, r29))
P("thresholds one step after the new maximum, theta_j(s+1) ~ a Mn + b M0 + c S0+ under L2 (units 1, 6, 7, 8, 4):")
for i, j in enumerate([1, 6, 7, 8, 4]):
    c, r2, _ = lsq(R["Xth"][R["L2"]], R["th"][R["L2"], i])
    P("   theta_%d: %.3f Mn + %.3f M0 + %.3f S0+   R2 %.4f" % (j, *c, r2))
P("   same under L1 only:")
for i, j in enumerate([1, 6, 7, 8, 4]):
    c, r2, _ = lsq(R["Xth"][R["L1"]], R["th"][R["L1"], i])
    P("   theta_%d: %.3f Mn + %.3f M0 + %.3f S0+   R2 %.4f" % (j, *c, r2))

# ============================================================ part 2: relabel sequences, single-step clamps
P("\n=== part 2: sequences whose final max (at s) overtook the true runner-up (old max at p0), g = s - p0 >= 2 (seeds 7044, 7045)")
res = {}


def radd(k, num, den):
    a = res.setdefault(k, [0, 0]); a[0] += num; a[1] += den


by = {k: [] for k in ("ex", "ex_cl", "act", "act_cl", "prov", "prov_cl", "exq", "actq", "s", "g")}
for sd in (7044, 7045):
    x = gauss(N, sd)
    st = prefix_stats(x)
    pM, pS = st["pM"][:, -1], st["pS"][:, -1]
    sel = (pS < pM) & (st["S"][:, -1] > 0) & (pM - pS >= 2)
    x = x[sel]; s = pM[sel]; p0 = pS[sel]
    B = len(x); rows = np.arange(B)
    stp = prefix_stats(x)
    q0 = prev(stp["pS"], -1)[rows, s]                    # old runner-up's position (before s)
    S0 = prev(stp["S"], -np.inf)[rows, s]
    H, Z = fwd(x)
    lg = H[:, -1] @ U.T
    ans = lg.argmax(1)
    radd("baseline P(p0)", (ans == p0).sum(), B)
    for g in range(2, 8):
        radd(f"baseline P(p0) g={g}", ((ans == p0) & (s - p0 == g)).sum(), (s - p0 == g).sum())
    # A: unit 5 off at p0+1 only
    HA, ZA = fwd(x, mask_hook([5], step_mask(B, p0 + 1), "zero"))
    lgA = HA[:, -1] @ U.T
    aA = lgA.argmax(1)
    radd("A unit5 off at p0+1: P(p0)", (aA == p0).sum(), B)
    radd("A: answers new max s", (aA == s).sum(), B); radd("A: answers old runner-up q0", (aA == q0).sum(), B)
    radd("A: answers p0+-1", (np.abs(aA - p0) == 1).sum(), B)
    radd("A: clamp changed unit 5 (it was on)", (Z[rows, p0 + 1, 5] > 0).sum(), B)
    radd("A: new max still gated as record (u4 on, comps off)", ((ZA[rows, s, 4] > 0) & (ZA[rows, s][:, COMP] <= 0).all(1)).sum(), B)
    g3 = (s - p0 >= 3) & (s >= 4)
    radd("A restricted to g>=3, s>=4: baseline", ((ans == p0) & g3).sum(), g3.sum())
    radd("A restricted to g>=3, s>=4: clamped", ((aA == p0) & g3).sum(), g3.sum())
    # same for unit 9 at p0+1 and unit 0 linear at p0+2
    a9 = (fwd(x, mask_hook([9], step_mask(B, p0 + 1), "zero"))[0][:, -1] @ U.T).argmax(1)
    radd("unit9 off at p0+1: P(p0)", (a9 == p0).sum(), B)
    a0 = (fwd(x, mask_hook([0], step_mask(B, p0 + 2), "linear"))[0][:, -1] @ U.T).argmax(1)
    radd("unit0 linear at p0+2: P(p0)", (a0 == p0).sum(), B)
    # B..E: clamps after the NEW max (s <= 8 resp. s <= 7)
    s8 = s <= 8
    radd("baseline P(p0), s<=8", ((ans == p0) & s8).sum(), s8.sum())
    for name, units, mode, off in (("B unit5 off at s+1", [5], "zero", 1), ("C unit9 off at s+1", [9], "zero", 1),
                                   ("D units 5,9 off at s+1", [5, 9], "zero", 1), ("E unit0 linear at s+2", [0], "linear", 2)):
        sub = s <= 9 - off
        aa = (fwd(x, mask_hook(units, step_mask(B, np.where(sub, s + off, -1)), mode))[0][:, -1] @ U.T).argmax(1)
        radd(name + ": P(p0)", ((aa == p0) & sub).sum(), sub.sum())
        radd(name + ": P(answer = new max s)", ((aa == s) & sub).sum(), sub.sum())
        radd(name + ": baseline P(p0) on same subset", ((ans == p0) & sub).sum(), sub.sum())
        radd(name + ": baseline P(answer = s) on same subset", ((ans == s) & sub).sum(), sub.sum())
    # ---- logits relative to bystanders (positions < s other than p0, q0); needs s >= 3
    pos = np.arange(10)[None, :]
    bys = (pos < s[:, None]) & (pos != p0[:, None]) & (pos != q0[:, None])
    hasb = bys.sum(1) >= 1
    xc = x.copy(); xc[rows, s] = np.minimum(S0, 0) - 1.0          # no-new-max counterfactual
    lgc = fwd(xc)[0][:, -1] @ U.T
    lgcA = fwd(xc, mask_hook([5], step_mask(B, p0 + 1), "zero"))[0][:, -1] @ U.T
    rel = lambda L, p: (L[rows, p] - (L * bys).sum(1) / np.maximum(bys.sum(1), 1))[hasb]
    by["ex"].append(rel(lgc, p0)); by["ex_cl"].append(rel(lgcA, p0)); by["act"].append(rel(lg, p0)); by["act_cl"].append(rel(lgA, p0))
    by["exq"].append(rel(lgc, q0)); by["actq"].append(rel(lg, q0))
    # provisional scores just before the new max: U h_{s-1}; index of position p is 9 - (s-1-p)
    def prov(Hh):
        sc = Hh[rows, s - 1] @ U.T                                # (B, 10) in age coordinates
        idx = np.clip(9 - (s[:, None] - 1 - pos), 0, 9)           # index of position p (valid for p <= s-1)
        scp = np.take_along_axis(sc, idx, 1)                      # (B, 10) by position p
        return (scp[rows, p0] - (scp * bys).sum(1) / np.maximum(bys.sum(1), 1))[hasb]
    by["prov"].append(prov(H)); by["prov_cl"].append(prov(HA))
    by["s"].append(s[hasb]); by["g"].append((s - p0)[hasb])
for k, (a, b) in res.items():
    P("  %-62s %.4f   (n = %d)" % (k, a / b, b))
Bd = {k: np.concatenate(v) for k, v in by.items()}
P("logit of the old max p0 minus mean logit of bystander positions (n = %d with >= 1 bystander):" % len(Bd["ex"]))
P("   no-new-max counterfactual, final logits : %.2f   | with the unit-5 pulse at p0+1 removed: %.2f" % (Bd["ex"].mean(), Bd["ex_cl"].mean()))
P("   actual final logits                     : %.2f   | with the pulse removed: %.2f" % (Bd["act"].mean(), Bd["act_cl"].mean()))
P("   created by the new max (actual - counterfactual): %.2f | with the pulse removed: %.2f" % ((Bd["act"] - Bd["ex"]).mean(), (Bd["act_cl"] - Bd["ex_cl"]).mean()))
P("   provisional scores U h_(s-1) just before the new max: %.2f | with the pulse removed: %.2f" % (Bd["prov"].mean(), Bd["prov_cl"].mean()))
P("   old runner-up q0: counterfactual %.2f, actual %.2f" % (Bd["exq"].mean(), Bd["actq"].mean()))

# ============================================================ part 3: clamps after EVERY positive new maximum
P("\n=== part 3: interventions after every positive new maximum (s = 0 included), all sequences (seeds 7046, 7047)")
tot = {}
for sd in (7046, 7047):
    x = gauss(N, sd)
    st = prefix_stats(x)
    rec = (x > prev(st["M"], -np.inf)) & (x > 0)                 # (B, 10) record steps
    pM, pS = st["pM"][:, -1], st["pS"][:, -1]
    ans = (fwd(x)[0][:, -1] @ U.T).argmax(1)
    m1 = np.zeros_like(rec); m1[:, 1:] = rec[:, :-1]
    m2 = np.zeros_like(rec); m2[:, 2:] = rec[:, :-2]
    for name, units, mask, mode in (("unit 5 off at s+1", [5], m1, "zero"), ("unit 9 off at s+1", [9], m1, "zero"),
                                    ("units 5 and 9 off at s+1", [5, 9], m1, "zero"), ("unit 0 linear at s+2", [0], m2, "linear"),
                                    ("unit 0 linear at s+1 (control)", [0], m1, "linear")):
        aa = (fwd(x, mask_hook(units, mask, mode))[0][:, -1] @ U.T).argmax(1)
        ch = aa != ans
        t = tot.setdefault(name, np.zeros(6))
        t += [len(x), (aa == ans).sum(), (aa == pS).sum(), ch.sum(), (ch & (aa == pM)).sum(), (ans == pS).sum()]
    P("  records per sequence: %.3f" % rec.sum(1).mean())
for name, t in tot.items():
    P("  %-32s agreement %.4f  accuracy %.4f (baseline %.4f)  share of changed answers that are the max %.4f" % (name, t[1] / t[0], t[2] / t[0], t[5] / t[0], t[4] / t[3]))
out.close()
