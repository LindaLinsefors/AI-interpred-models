"""Verifier V1, attack E3: is the 61.6 slope a property of the readout or of the data distribution?

(a) weights only: k(i) = (U[9] - U[i]) . (g * w) for the gate pattern g = "13 ordinary units on, pulses off"
(b) data: exact derivative d(s_t[9] - s_t[idx p2]) / d x_t = (U[9] - U[idx]) . (G_t * w) for every age-0
    "other" entry (S_t > 0.2, t = 2..9), its distribution, by comparator gate pattern; and the tie point
    x* of the local linear piece compared with S_t.
(c) regression of s[9] - s[p2] on (x_t, S_t, M_t) without intercept.
(d) other input distributions (same statistic as T4a).
(e) sweep of x_t at fixed history: per-state slope over v in [-2,-0.05] and the exact tie point.
Seeds 5501.. (stated per part).  Run: python -B v1_attack_slope.py -> v1_attack_slope.txt
"""
from vlib import *

P = Tee("v1_attack_slope.txt")

# ---------------------------------------------------------------- (a)
P("=== (a) weights only ===")
g_rest = np.zeros(16); g_rest[ORD] = 1
kk = [(U[9] - U[i]) @ (g_rest * w) for i in range(9)]
P("k(i) = (U[9]-U[i]).w over the 13 ordinary units, i = 0..8:", np.round(kk, 2), " mean %.2f  min %.2f  max %.2f" % (np.mean(kk), np.min(kk), np.max(kk)))
g_c = np.zeros(16); g_c[COMP] = 1
P("  comparators only:", np.round([(U[9] - U[i]) @ (g_c * w) for i in range(9)], 2))
P("  U[i,COMP].w[COMP], i=0..9:", np.round(U[:, COMP] @ w[COMP], 2))
P("  U[i,ORD].w[ORD],  i=0..9:", np.round(U[:, ORD] @ w[ORD], 2))
P("  U[i,WAVE].w[WAVE], i=0..9:", np.round(U[:, WAVE] @ w[WAVE], 2))
P("  per-unit U[9,u]*w[u] - mean_i<9 U[i,u]*w[u]:", {u: round(float((U[9, u] - U[:9, u].mean()) * w[u]), 2) for u in range(16)})


# ---------------------------------------------------------------- helpers
def age0_table(x, thr=0.2):
    """age-0 'other' entries: returns dict of arrays over (state) with x_t not in the top two of x_0..x_t."""
    H, Z = run(x, return_pre=True)
    out = {k: [] for k in ("d", "v", "xt", "S", "M", "T3", "t", "a2", "g", "c0", "gc")}
    for t in range(2, 10):
        sub, full = scores(H, t)
        p, val = top3(x, t)
        p1, p2, M, S = p[:, 0], p[:, 1], val[:, 0], val[:, 1]
        m = (S > thr) & (p1 != t) & (p2 != t)
        r = np.where(m)[0]
        idx = 9 - t + p2[m]
        G = (Z[m, t] > 0).astype(float)
        dU = U[9][None, :] - U[idx]                                   # (n, 16)
        g = (dU * G * w).sum(1)
        diff = full[m, 9] - full[r, idx]
        out["d"].append(diff / M[m]); out["v"].append((x[m, t] - S[m]) / M[m])
        out["xt"].append(x[m, t]); out["S"].append(S[m]); out["M"].append(M[m]); out["T3"].append(val[m, 2])
        out["t"].append(np.full(m.sum(), t)); out["a2"].append(t - p2[m]); out["g"].append(g)
        out["c0"].append(diff - g * x[m, t])
        out["gc"].append((G[:, COMP] * (1 << np.arange(4))).sum(1).astype(int))
    return {k: np.concatenate(v) for k, v in out.items()}


def t4a_stats(tb, label):
    m = (tb["v"] >= -2) & (tb["v"] <= -0.05)
    d, v = tb["d"][m], tb["v"][m]
    edges = np.array([-2, -1.5, -1, -0.7, -0.5, -0.35, -0.25, -0.15, -0.10, -0.05])
    b = np.searchsorted(edges, v, side="left") - 1
    rat = [np.median(d[b == k]) / np.median(v[b == k]) for k in range(len(edges) - 1)]
    fit = np.polyfit(v, d, 1)
    P(f"   {label:34s} n={m.sum():8d}  LS0 slope {(d*v).sum()/(v*v).sum():6.2f}  median(d/v) {np.median(d/v):6.2f}  LS slope+intercept {fit[0]:6.2f} {fit[1]:+5.2f}"
      f"  bin ratios {np.round(rat, 1)}")


# ---------------------------------------------------------------- (b), (c)
P("\n=== (b) exact derivative on data, N(0,1), seeds 5501..5503 (200k each) ===")
tbs = [age0_table(sample(200_000, s)) for s in (5501, 5502, 5503)]
tb = {k: np.concatenate([q[k] for q in tbs]) for k in tbs[0]}
m = (tb["v"] >= -2) & (tb["v"] <= -0.05)
P("entries (age-0 other, S>0.2):", len(tb["d"]), "; with v in [-2,-0.05]:", int(m.sum()))
t4a_stats(tb, "N(0,1) (reference, = T4a)")
g = tb["g"]
P("derivative g = d(s[9]-s[p2])/dx_t :  mean %.2f  median %.2f  sd %.2f  percentiles 1,5,25,75,95,99: %s"
  % (g[m].mean(), np.median(g[m]), g[m].std(), np.round(np.percentile(g[m], [1, 5, 25, 75, 95, 99]), 2)))
P("by comparator gate pattern at t (bit order 1,6,7,8 ; 15 = all four on), v in [-2,-0.05]:")
for pat in np.unique(tb["gc"][m]):
    mm = m & (tb["gc"] == pat)
    P(f"   pattern {pat:2d} ({''.join(str((pat >> k) & 1) for k in range(4))}): share {mm.sum()/m.sum():.4f}  median g {np.median(g[mm]):7.2f}  sd {g[mm].std():5.2f}  median d/v {np.median(tb['d'][mm]/tb['v'][mm]):7.2f}")
P("same for ALL v<0 (incl. v > -0.05):")
m_all = tb["v"] < 0
for pat in np.unique(tb["gc"][m_all]):
    mm = m_all & (tb["gc"] == pat)
    P(f"   pattern {pat:2d}: share {mm.sum()/m_all.sum():.4f}  median g {np.median(g[mm]):7.2f}   median v {np.median(tb['v'][mm]):7.3f}")
P("by age of the runner-up a2 (all four comparators on): median g:",
  {int(a): round(float(np.median(g[m & (tb['gc'] == 15) & (tb['a2'] == a)])), 2) for a in range(1, 10)})
# tie point of the local linear piece:  s[9]-s[p2] = g x_t + c0  -> x* = -c0/g
xs = -tb["c0"] / g
e = (xs - tb["S"]) / tb["M"]
P("tie point x* of the local linear piece (gates of the state held fixed): (x* - S)/M  median %.4f  mean %.4f  sd %.4f  pct 5,25,75,95: %s"
  % (np.median(e[m]), e[m].mean(), e[m].std(), np.round(np.percentile(e[m], [5, 25, 75, 95]), 4)))
P("   by t: " + "  ".join(f"t{t}: {np.median(e[m & (tb['t']==t)]):+.4f}/{e[m & (tb['t']==t)].std():.3f}" for t in range(2, 10)))
P("   by a2: " + "  ".join(f"a{a}: {np.median(e[m & (tb['a2']==a)]):+.4f}/{e[m & (tb['a2']==a)].std():.3f}" for a in range(1, 10)))
resid = tb["d"][m] - 61.6 * tb["v"][m]
P("residual d - 61.6 v on [-2,-0.05]: mean %.3f  median %.3f  sd %.3f   (sd/61.6 = %.4f in units of M)" % (resid.mean(), np.median(resid), resid.std(), resid.std() / 61.6))

P("\n=== (c) regression of s[9]-s[p2] (raw logits, not divided by M) on (x_t, S_t, M_t), no intercept; v in [-2,-0.05] ===")
y = tb["d"][m] * tb["M"][m]
X = np.c_[tb["xt"][m], tb["S"][m], tb["M"][m]]
co, *_ = np.linalg.lstsq(X, y, rcond=None)
P("   coefficients (x_t, S, M):", np.round(co, 2), " uncentred R2 %.4f" % (1 - ((y - X @ co) ** 2).sum() / (y ** 2).sum()),
  " centred R2 %.4f" % (1 - ((y - X @ co) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
T3p = np.maximum(tb["T3"][m], 0)
X2 = np.c_[X, T3p]
co2, *_ = np.linalg.lstsq(X2, y, rcond=None)
P("   with max(third,0) added (x_t, S, M, T+):", np.round(co2, 2), " centred R2 %.4f" % (1 - ((y - X2 @ co2) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
for a in (1, 2, 3, 5, 7, 9):
    mm = tb["a2"][m] == a
    c_, *_ = np.linalg.lstsq(X[mm], y[mm], rcond=None)
    P(f"   a2={a}: (x_t, S, M) = {np.round(c_, 2)}")
co1 = (y * (tb["xt"][m] - tb["S"][m])).sum() / ((tb["xt"][m] - tb["S"][m]) ** 2).sum()
P("   one-parameter fit y = k (x_t - S): k = %.2f, centred R2 %.4f" % (co1, 1 - ((y - co1 * (tb["xt"][m] - tb["S"][m])) ** 2).sum() / ((y - y.mean()) ** 2).sum()))

# ---------------------------------------------------------------- (d)
P("\n=== (d) other input distributions (200k sequences each, rng seeds 5511..5517); T4a statistic ===")
rng = np.random.default_rng
n = 200_000
dists = {
    "N(0,1)  seed 5511": rng(5511).standard_normal((n, 10)),
    "Unif(-1,1)  seed 5512": rng(5512).uniform(-1, 1, (n, 10)),
    "Laplace(0,1)  seed 5513": rng(5513).laplace(0, 1, (n, 10)),
    "N(1,1) (shifted up)  seed 5514": rng(5514).standard_normal((n, 10)) + 1,
    "Exp(1)-0.5 (skewed)  seed 5515": rng(5515).exponential(1, (n, 10)) - 0.5,
    "Student-t(3)  seed 5516": rng(5516).standard_t(3, (n, 10)),
    "AR(1) rho=0.8 Gaussian  seed 5517": None,
}
e_ = rng(5517).standard_normal((n, 10)); ar = np.empty((n, 10)); ar[:, 0] = e_[:, 0]
for t in range(1, 10):
    ar[:, t] = 0.8 * ar[:, t - 1] + 0.6 * e_[:, t]
dists["AR(1) rho=0.8 Gaussian  seed 5517"] = ar
for name, xx in dists.items():
    tb_ = age0_table(xx)
    t4a_stats(tb_, name)

# ---------------------------------------------------------------- (e)
P("\n=== (e) sweep of x_t at fixed history (N(0,1) histories, seed 5521, 60k sequences; one random t in 2..9 per sequence) ===")
x = sample(60_000, 5521)
r5 = np.random.default_rng(5522)
tt = r5.integers(2, 10, len(x))
H = run(x)
grid = np.r_[np.linspace(-2, -0.05, 14), np.linspace(-0.045, 0.2, 50)]
slopes, ties, a2s, ts, n_used = [], [], [], [], 0
for t in range(2, 10):
    sel = np.where(tt == t)[0]
    p, val = top3(x[sel], t - 1)
    p2, M, S = p[:, 1], val[:, 0], val[:, 1]
    ok = S > 0.2
    sel, p2, M, S = sel[ok], p2[ok], M[ok], S[ok]
    h0 = H[sel, t - 1]                                    # (n, 16)
    idx = 9 - t + p2
    r = np.arange(len(sel))
    D = np.empty((len(sel), len(grid)))
    for j, vv in enumerate(grid):
        xt = S + vv * M
        h = np.maximum(xt[:, None] * w + h0 @ W.T, 0)
        s = h @ U.T
        D[:, j] = (s[:, 9] - s[r, idx]) / M
    lin = grid <= -0.05
    gv = grid[lin]
    sl = ((D[:, lin] - D[:, lin].mean(1, keepdims=True)) * (gv - gv.mean())).sum(1) / ((gv - gv.mean()) ** 2).sum()
    # tie point: first sign change of D along the fine grid (linear interpolation)
    fine = np.where(grid >= -0.2)[0]
    Df = D[:, fine]; gf = grid[fine]
    pos = Df > 0
    first = pos.argmax(1)
    has = pos.any(1) & (first > 0)
    tie = np.full(len(sel), np.nan)
    i1 = first[has]; rr = np.where(has)[0]
    y0, y1 = Df[rr, i1 - 1], Df[rr, i1]
    tie[has] = gf[i1 - 1] + (gf[i1] - gf[i1 - 1]) * (-y0) / (y1 - y0)
    slopes.append(sl); ties.append(tie); a2s.append(t - 1 - p2 + 1); ts.append(np.full(len(sel), t)); n_used += len(sel)
slopes, ties, a2s, ts = map(np.concatenate, (slopes, ties, a2s, ts))
P(f"states used: {n_used}")
P("per-state LS slope of d on v over v in [-2,-0.05]: median %.2f  mean %.2f  sd %.2f  pct 1,5,25,75,95,99: %s"
  % (np.median(slopes), slopes.mean(), slopes.std(), np.round(np.percentile(slopes, [1, 5, 25, 75, 95, 99]), 2)))
P("   share of states with slope within 61.6 +- 2: %.3f ; +- 5: %.3f" % (np.mean(np.abs(slopes - 61.6) < 2), np.mean(np.abs(slopes - 61.6) < 5)))
okk = ~np.isnan(ties)
P("tie point v* (s[9] = s[old runner-up]) in units of M, relative to the runner-up (0 = exact comparison by value):")
P("   found in %.4f of states on the grid [-0.2, 0.2];  median %+.4f  mean %+.4f  sd %.4f  pct 5,25,75,95: %s"
  % (okk.mean(), np.nanmedian(ties), np.nanmean(ties), np.nanstd(ties), np.round(np.nanpercentile(ties, [5, 25, 75, 95]), 4)))
P("   by t: " + "  ".join(f"t{t}: {np.nanmedian(ties[ts==t]):+.4f}" for t in range(2, 10)))
P("   by age of the runner-up at t: " + "  ".join(f"a{a}: {np.nanmedian(ties[a2s==a]):+.4f}" for a in range(1, 10)))
