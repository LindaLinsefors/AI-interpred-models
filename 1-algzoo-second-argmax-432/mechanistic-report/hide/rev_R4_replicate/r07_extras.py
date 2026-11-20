"""R4 / r07: follow-ups that explain small differences found in r02-r06.
 (1) unit 9 / unit 0 / unit 5 clamps after the final max: dependence on the subset of sequences
 (2) the old max's logit relative to bystanders on arm F's subsets
 (3) unit 0 off at s+2: dependence on what follows the new maximum
 (4) size of the 'small' remainder in Theta_5 = 0.01265 m_runner - 0.02872 m_level + small
 (5) unit 5 forced off only after inputs with 0.9 <= x/S < 1
Seeds 7081-7086.  Output: r07_extras_out.txt
"""
from rlib import *
import modes

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


def mask_hook(units, mask, mode):
    def hook(t, z, h):
        m = mask[:, t]
        if m.any():
            for u in units:
                h[m, u] = 0.0 if mode == "zero" else z[m, u]
        return h
    return hook


def step_mask(B, step):
    m = np.zeros((B, 10), bool)
    ok = (step >= 0) & (step <= 9)
    m[np.nonzero(ok)[0], step[ok]] = True
    return m


N = 400_000
acc = {}


def radd(k, num, den):
    a = acc.setdefault(k, [0, 0]); a[0] += num; a[1] += den


lgs = {k: [] for k in ("ex", "ex_cl", "act", "act_cl", "exq", "actq", "s", "g", "S0")}
for sd in (7081, 7082):
    x = gauss(N, sd)
    st = prefix_stats(x)
    pM, pS = st["pM"][:, -1], st["pS"][:, -1]
    rows = np.arange(N)
    ans = (fwd(x)[0][:, -1] @ U.T).argmax(1)
    S0all = prev(st["S"], -np.inf)[rows, pM]                 # old runner-up before the final max
    # ---- (1) clamps after the FINAL max, several subsets (all need s <= 8, resp. s <= 7)
    subsets = {
        "all sequences, M>0": st["M"][:, -1] > 0,
        "runner-up before max (any gap g>=1), S>0": (pS < pM) & (st["S"][:, -1] > 0),
        "g>=2, S>0 (r03)": (pS < pM) & (st["S"][:, -1] > 0) & (pM - pS >= 2),
        "g>=2, S>0, s>=3 (arm F)": (pS < pM) & (st["S"][:, -1] > 0) & (pM - pS >= 2) & (pM >= 3),
        "g>=2, S>0, s==2": (pS < pM) & (st["S"][:, -1] > 0) & (pM - pS >= 2) & (pM == 2),
        "runner-up after max, S>0": (pS > pM) & (st["S"][:, -1] > 0),
    }
    for name, units, mode, off in (("unit 9 off at s+1", [9], "zero", 1), ("unit 5 off at s+1", [5], "zero", 1),
                                   ("units 5,9 off at s+1", [5, 9], "zero", 1), ("unit 0 linear at s+2", [0], "linear", 2)):
        aa = (fwd(x, mask_hook(units, step_mask(N, np.where(pM <= 9 - off, pM + off, -1)), mode))[0][:, -1] @ U.T).argmax(1)
        for sn, sm in subsets.items():
            m = sm & (pM <= 9 - off)
            radd(f"{name} | {sn} | P(correct)", ((aa == pS) & m).sum(), m.sum())
            radd(f"{name} | {sn} | P(answer = max)", ((aa == pM) & m).sum(), m.sum())
            radd(f"{name} | {sn} | baseline P(correct)", ((ans == pS) & m).sum(), m.sum())
            radd(f"{name} | {sn} | baseline P(answer = max)", ((ans == pM) & m).sum(), m.sum())
    # ---- (2) old max's logit relative to bystanders
    sel = (pS < pM) & (st["S"][:, -1] > 0) & (pM - pS >= 2) & (pM >= 3)
    xs = x[sel]; s = pM[sel]; p0 = pS[sel]; B = len(xs); r = np.arange(B)
    sp = prefix_stats(xs)
    q0 = prev(sp["pS"], -1)[r, s]; S0 = prev(sp["S"], -np.inf)[r, s]
    pos = np.arange(10)[None, :]
    bys = (pos < s[:, None]) & (pos != p0[:, None]) & (pos != q0[:, None])
    xc = xs.copy(); xc[r, s] = np.minimum(S0, 0) - 1.0
    hk = mask_hook([5], step_mask(B, p0 + 1), "zero")
    L = {"act": fwd(xs)[0][:, -1] @ U.T, "act_cl": fwd(xs, hk)[0][:, -1] @ U.T,
         "ex": fwd(xc)[0][:, -1] @ U.T, "ex_cl": fwd(xc, hk)[0][:, -1] @ U.T}
    rel = lambda A, p: A[r, p] - (A * bys).sum(1) / bys.sum(1)
    for k in ("act", "act_cl", "ex", "ex_cl"):
        lgs[k].append(rel(L[k], p0))
    lgs["exq"].append(rel(L["ex"], q0)); lgs["actq"].append(rel(L["act"], q0))
    lgs["s"].append(s); lgs["g"].append(s - p0); lgs["S0"].append(S0)

P("=== (1) single clamps after the final maximum (at s), 2 x 400k sequences (seeds 7081-7082)")
keys = sorted({k.rsplit(" | ", 1)[0] for k in acc})
for k in keys:
    g = lambda q: acc[f"{k} | {q}"][0] / acc[f"{k} | {q}"][1]
    P("  %-64s n=%7d  P(correct) %.4f (baseline %.4f)  P(answer = max) %.4f (baseline %.4f)" % (
        k, acc[f"{k} | P(correct)"][1], g("P(correct)"), g("baseline P(correct)"), g("P(answer = max)"), g("baseline P(answer = max)")))

Ld = {k: np.concatenate(v) for k, v in lgs.items()}
P("\n=== (2) logit of the old max p0 (and old runner-up q0) minus the mean over bystander positions; final max at s>=3, g>=2")
for name, m in (("all (S0 any sign)", np.ones(len(Ld["s"]), bool)), ("old runner-up S0 > 0 (arm F, s06c)", Ld["S0"] > 0),
                ("g >= 3 and s >= 4 (arm F, s05c)", (Ld["g"] >= 3) & (Ld["s"] >= 4))):
    P("  %-36s n=%6d  p0: without new max %.2f -> actual %.2f (created %.2f) ; q0: %.2f -> %.2f ; with the old pulse removed: p0 without new max %.2f, actual %.2f (created %.2f)" % (
        name, m.sum(), Ld["ex"][m].mean(), Ld["act"][m].mean(), (Ld["act"] - Ld["ex"])[m].mean(), Ld["exq"][m].mean(), Ld["actq"][m].mean(),
        Ld["ex_cl"][m].mean(), Ld["act_cl"][m].mean(), (Ld["act_cl"] - Ld["ex_cl"])[m].mean()))

# ---- (3) unit 0 at s+2 by what follows; (4) Theta5 remainder
P("\n=== (3) unit 0 OFF at s+2 after a new maximum at s = 1..7 with M0 > 0, by what follows (seed 7083)")
x = gauss(N, 7083)
H, Z = fwd(x)
st = prefix_stats(x)
Mp, Sp = prev(st["M"], -np.inf), prev(st["S"], -np.inf)
tab = {}
for s in range(1, 8):
    rec = (x[:, s] > Mp[:, s]) & (Mp[:, s] > 0)
    Mn, M0 = x[:, s], Mp[:, s]
    a, b = x[:, s + 1], x[:, s + 2]
    S0p = np.maximum(Sp[:, s], 0)
    conds = {
        "all new maxima": rec,
        "no newer max at s+1, s+2 (both < Mn)": rec & (a < Mn) & (b < Mn),
        "x_(s+1), x_(s+2) < M0 (below the new runner-up)": rec & (a < M0) & (b < M0),
        "x_(s+1), x_(s+2) < old runner-up S0+": rec & (a < S0p) & (b < S0p),
        "x_(s+1) > Mn (another new max at once)": rec & (a > Mn),
        "M0 < x_(s+1) < Mn (new runner-up at s+1)": rec & (a > M0) & (a < Mn),
        "x_(s+1) < M0, x_(s+2) > Mn": rec & (a < M0) & (b > Mn),
        "x_(s+1) < M0, M0 < x_(s+2) < Mn": rec & (a < M0) & (b > M0) & (b < Mn),
    }
    for k, m in conds.items():
        t = tab.setdefault(k, np.zeros(2)); t += [m.sum(), (Z[m, s + 2, 0] <= 0).sum()]
tot = tab["all new maxima"][0]
for k, t in tab.items():
    P("  %-52s share %.3f  P(unit 0 off at s+2) %.4f" % (k, t[0] / tot, t[1] / t[0]))

P("\n=== (4) Theta_5 against 0.01265 m_runner - 0.02872 m_level (coordinates of h_(t-1)), steps t = 2..8 with S > 0 (seed 7083)")
c5 = W[5, ORD] @ w[ORD]
rem_all, th_all, M_all, c78_all, u4_all, gap_all = [], [], [], [], [], []
for t in range(2, 9):
    pre = H[:, t - 1] @ W.T
    th = -(pre[:, ORD] @ W[5, ORD] + H[:, t][:, PULSE] @ W[5, PULSE]) / c5
    m = modes.coords(H[:, t - 1])
    two = 0.01265 * m["runner"] - 0.02872 * m["level"]
    ok = Sp[:, t] > 0
    rem_all.append((th - two)[ok]); th_all.append(th[ok]); M_all.append(Mp[ok, t])
    c78_all.append(((Z[:, t, 7] > 0) & (Z[:, t, 8] > 0))[ok]); u4_all.append((Z[:, t, 4] > 0)[ok])
rem, th, Mv, c78, u4 = map(np.concatenate, (rem_all, th_all, M_all, c78_all, u4_all))
for name, m in (("comparators 7, 8 on at t (the formula's condition)", c78), ("  of which unit 4 silent", c78 & ~u4), ("  of which unit 4 fires", c78 & u4)):
    q = np.percentile(np.abs(rem[m]) / Mv[m], [50, 90, 99])
    P("  %-52s n=%7d  |remainder|/M: median %.3f, 90%% %.3f, 99%% %.3f ; Theta5/M median %.3f ; R2 of the two-term form %.3f" % (
        name, m.sum(), *q, np.median(th[m] / Mv[m]), 1 - (rem[m] ** 2).sum() / ((th[m] - th[m].mean()) ** 2).sum()))

# ---- (5) unit 5 off after inputs below the runner-up, by x/S band ; and share of inputs after which unit 5 fires
P("\n=== (5) unit 5 := 0 at t+1 after inputs below the runner-up (seeds 7084-7086, 3 x 400k)")
tt = {}
ev = np.zeros((4, 2))
for sd in (7084, 7085, 7086):
    x = gauss(N, sd)
    st = prefix_stats(x)
    Mp, Sp = prev(st["M"], -np.inf), prev(st["S"], -np.inf)
    pS = st["pS"][:, -1]
    H, Z = fwd(x)
    ans = (H[:, -1] @ U.T).argmax(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = x / Sp
    below = (x < Sp) & np.isfinite(Sp)
    bands = {"none": np.zeros_like(below), "every x_t < S": below, "S>0, 0.9 <= x/S < 1": below & (Sp > 0) & (ratio >= 0.9),
             "S>0, 0.5 <= x/S < 0.9": below & (Sp > 0) & (ratio >= 0.5) & (ratio < 0.9), "S>0, x/S < 0.5": below & (Sp > 0) & (ratio < 0.5)}
    for k, b in bands.items():
        mk = np.zeros_like(b); mk[:, 1:] = b[:, :-1]
        aa = (fwd(x, mask_hook([5], mk, "zero"))[0][:, -1] @ U.T).argmax(1)
        t = tt.setdefault(k, np.zeros(3)); t += [N, (aa == pS).sum(), (aa == ans).sum()]
    on5 = np.zeros_like(below); on5[:, :-1] = Z[:, 1:, 5] > 0             # unit 5 on at t+1
    newmax = (x > Mp) & (x > 0); newsec = (x > Sp) & (x < Mp) & (x > 0)
    for i, m in enumerate((newmax[:, :9], newsec[:, :9], below[:, :9], (below & (x < 0))[:, :9])):
        ev[i] += [m.sum(), (m & on5[:, :9]).sum()]
for k, t in tt.items():
    P("  %-26s accuracy %.4f  agreement %.4f" % (k, t[1] / t[0], t[2] / t[0]))
P("  unit 5 on at t+1 after: positive new max %.4f ; positive new runner-up %.4f ; input below the runner-up %.4f ; negative input below the runner-up %.4f" % tuple(ev[:, 1] / ev[:, 0]))
out.close()
