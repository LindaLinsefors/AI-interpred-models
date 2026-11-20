"""I4 (checks on the takeover offsets of i04b).
 (1) Cross-validated R^2 of the cell means (p_e, p_l, p_M): fit on one half, score on the other.
 (2) A less restricted sample (M - S >= 0.1 M, fourth < S - 0.06 M, S > 0.1 M), sweep +-0.05 M: same distribution?
 (3) Do the sweeps predict the census?  P(error | S~T) from eps*:  S first: T wins iff -gap > eps*;  T first: T wins iff gap < eps*.
 (4) Three-value inputs on a constant background reproduce the cell medians (so the offsets are a property of the event
     configuration, not of the other inputs)."""
from icommon import *
import i04b_takeover as tk

Z = np.load(HIDE / "r1_I_errors" / "i04b_RT.npz")
eps, pe, pl, pM, M, S = (Z[k] for k in ("eps", "pe", "pl", "pM", "M", "S"))
fin = np.isfinite(eps)
key = (pe * 100 + pl * 10 + pM)

# (1)
rng = np.random.default_rng(1080)
half = rng.random(len(eps)) < 0.5
tr = fin & half; te = fin & ~half
mu = {}
for kk in np.unique(key[tr]):
    mu[int(kk)] = eps[tr & (key == kk)].mean()
pred = np.array([mu.get(int(k), eps[tr].mean()) for k in key[te]])
v = eps[te]
print(f"(1) cross-validated R^2 of (p_e, p_l, p_M) cell means: {1 - ((v - pred)**2).sum() / ((v - v.mean())**2).sum():.3f}  (train {tr.sum()}, test {te.sum()})")
med = {}
for kk in np.unique(key[tr]):
    med[int(kk)] = np.median(eps[tr & (key == kk)])
predm = np.array([med.get(int(k), 0.0) for k in key[te]])
print(f"    P(sign of eps* = sign of the cell median from the other half) = {np.mean(np.sign(v) == np.sign(predm)):.3f}")

# (2)
x = sample(2 ** 18, 1081)
p_M, p_S, p_T, Mv, Sv, Tv = top3(x)
fourth = np.sort(x, 1)[:, -4]
sel = (Mv - Sv >= 0.1 * Mv) & (fourth < Sv - 0.06 * Mv) & (Sv > 0.1 * Mv)
idx = np.flatnonzero(sel)[:60000]
x, p_M, p_S, p_T, Mv, Sv = (a[idx] for a in (x, p_M, p_S, p_T, Mv, Sv))
e_, l_ = np.minimum(p_S, p_T), np.maximum(p_S, p_T)
xs = x.copy(); r = np.arange(len(x)); xs[r, e_] = Sv
old = tk.EPS
tk.EPS = np.round(np.arange(-0.05, 0.05001, 0.0025), 5)
D, A = tk.sweep(xs, e_, l_, "RT", Mv)
eps2, nch2 = tk.crossing(D)
tk.EPS = old
f2 = np.isfinite(eps2)
print(f"\n(2) less restricted sample (seed 1081, n = {len(x)}; share of all sequences that qualify {sel.mean():.3f}), sweep +-0.05 M:")
print(f"    crossing inside +-0.05: {f2.mean():.4f}; below {np.mean(eps2 == -np.inf):.4f}; above {np.mean(eps2 == np.inf):.4f}")
# quantiles treating out-of-window values as beyond the window
e2 = np.where(eps2 == -np.inf, -1, np.where(eps2 == np.inf, 1, eps2))
print("    eps* quantiles 10/25/50/75/90%: " + " / ".join(f"{q:+.4f}" for q in np.quantile(e2, [.1, .25, .5, .75, .9])) + "   (restricted sample of i04b: "
      + " / ".join(f"{q:+.4f}" for q in np.quantile(eps[fin], [.1, .25, .5, .75, .9])) + ")")

# (3) predicted error rates against the census (class T > 0, g1 >= 0.1), using the less restricted sample
print("\n(3) census error rate of the S~T comparison against the prediction from the sweeps (less restricted sample)")
Zc = np.load(HIDE / "r1_I_errors" / "i01_census.npz")
cM, cS, cT = (Zc[k].astype(np.float64) for k in ("M", "S", "T"))
cok = Zc["ok"]; cpS, cpT, crank = Zc["pS"], Zc["pT"], Zc["rank"]
g2 = (cS - cT) / cM; g1 = (cM - cS) / cM
base = (cT > 0) & (g1 >= 0.1)
rows = []
for lo, hi in ((0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05)):
    mid = (lo + hi) / 2
    c1 = base & (g2 >= lo) & (g2 < hi) & (cpS < cpT); c2 = base & (g2 >= lo) & (g2 < hi) & (cpS > cpT)
    obs1 = np.mean(~cok[c1] & (crank[c1] == 3)); obs2 = np.mean(~cok[c2] & (crank[c2] == 3))
    pr1 = np.mean(e2 < -mid); pr2 = np.mean(e2 > mid)
    rows.append([f"[{lo}, {hi})", f"{obs1:.3f}", f"{pr1:.3f}", f"{obs2:.3f}", f"{pr2:.3f}"])
print(fmt_table(rows, ["g2 = (S-T)/M", "S first: observed P(answer = third)", "predicted P(eps* < -g2)", "T first: observed", "predicted P(eps* > g2)"]))

# (4) three-value inputs on a constant background
print("\n(4) three-value inputs (M = 1 at p_M, S-level 0.5 at p_e, later candidate swept at p_l, background b): eps* against the cell medians of the N(0,1) sweeps")
cells = [(0, 7, 9), (0, 8, 9), (0, 9, 4), (0, 9, 6), (3, 4, 0), (8, 9, 6), (1, 7, 0), (4, 6, 7), (2, 7, 0), (7, 8, 9), (3, 5, 0), (1, 4, 0), (2, 5, 8), (0, 3, 1)]
rows = []
for (a, b_, c) in cells:
    kk = a * 100 + b_ * 10 + c
    s = fin & (key == kk)
    row = [a, b_, c, f"{np.median(eps[s]):+.4f}" if s.sum() >= 30 else "-", int(s.sum())]
    for bg in (-1.0, -0.3):
        xx = np.full((1, 10), bg); xx[0, c] = 1.0; xx[0, a] = 0.5
        Dd, _ = tk.sweep(xx, np.array([a]), np.array([b_]), "RT", np.array([1.0]))
        ee, _ = tk.crossing(Dd)
        row.append(f"{ee[0]:+.4f}")
    rows.append(row)
print(fmt_table(rows, ["p_e", "p_l", "p_M", "median eps*, N(0,1) sweeps", "n", "three-value input, b = -1", "b = -0.3"]))
allc = []
for a in range(10):
    for b_ in range(a + 1, 10):
        for c in range(10):
            if c in (a, b_): continue
            kk = a * 100 + b_ * 10 + c; s = fin & (key == kk)
            if s.sum() < 80: continue
            xx = np.full((1, 10), -1.0); xx[0, c] = 1.0; xx[0, a] = 0.5
            Dd, _ = tk.sweep(xx, np.array([a]), np.array([b_]), "RT", np.array([1.0]))
            ee, _ = tk.crossing(Dd)
            if np.isfinite(ee[0]): allc.append((np.median(eps[s]), ee[0]))
allc = np.array(allc)
print(f"over {len(allc)} cells with n >= 80: correlation between the cell median (N(0,1) sweeps) and the three-value eps* (b = -1, S/M = 0.5): {np.corrcoef(allc.T)[0, 1]:.3f}; "
      f"median absolute difference {np.median(np.abs(allc[:, 0] - allc[:, 1])):.4f}")
