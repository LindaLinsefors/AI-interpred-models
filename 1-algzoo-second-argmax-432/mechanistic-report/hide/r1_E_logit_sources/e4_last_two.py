"""E4. How the last two positions (9 = current input, 8 = previous input) are scored.

Part A [EXACT]   closed form of all logits at the last step in terms of x_9, the thresholds theta_j(9) and the stored
                 drive b = W h_8; slopes in each regime.
Part B           sweep x_9 for fixed prefixes (seed 1040, n = 20000): when does the network answer 9?  Where are the two
                 edges, and which stored position is the competitor at each edge?
Part C           same for position 8 (sweep x_8, the other nine inputs fixed; seed 1041).
Part D [INTERVENTION]  zero single readout weights U[9,4], U[9,7], U[8,5], U[8,9] and measure accuracy by case
                 (seed 1042, n = 400k).
"""
from src import *

np.set_printoptions(precision=2, suppress=True, linewidth=220)
REST = WAVE + [5, 9]

# ------------------------------------------------------------------------------------------------ Part A
print("=== Part A [EXACT] ===")
print("L_p = sum_{j in COMP} U[p,j]|w_j| (theta_j - x_9)_+  +  U[p,4] w_4 (x_9 - theta_4)_+  +  sum_{i in WAVE+{5,9}} U[p,i] (b_i + w_i x_9)_+")
print("coefficients U[p,j]|w_j| (rows: position p; columns: comparators 1, 6, 7, 8 | sum of the four | U[p,4] w_4):")
for p in range(10):
    cf = [U[p, j] * abs(w[j]) for j in COMP]
    print(f"  p{p}: " + " ".join(f"{v:8.2f}" for v in cf) + f" | {sum(cf):8.2f} | {U[p, 4] * w[4]:8.2f}")
slope_low = np.array([sum(U[p, i] * w[i] for i in ORD) for p in range(10)])
print("slope dL_p/dx_9 with all 13 ordinary units on, unit 4 off (= KX(0)):", slope_low)
print("   -> d(L_9 - L_p)/dx_9 for p = 0..8:", slope_low[9] - slope_low[:9])
wave_part = np.array([sum(U[p, i] * w[i] for i in WAVE) for p in range(10)])
slope_hi = U[:, 4] * w[4] + wave_part
print("slope dL_p/dx_9 with the four comparators clipped, unit 4 on, wave units on:", slope_hi)
print("   -> d(L_9 - L_p)/dx_9 for p = 0..8:", slope_hi[9] - slope_hi[:9])
print("each comparator j, while ON, adds (U[9,j] - U[p,j]) w_j to d(L_9 - L_p)/dx_9 (rows p = 0..8, columns j = 1,6,7,8):")
for p in range(9):
    print(f"  p{p}: " + " ".join(f"{(U[9, j] - U[p, j]) * w[j]:8.2f}" for j in COMP))
print("unit 5 at the last step: h5_9 = (W[5].h_8 + w_5 x_9)_+ ;  W[5,7]|w_7| = %.2f, W[5,8]|w_8| = %.2f, W[5,4] w_4 = %.2f, w_5 = %.2f"
      % (W[5, 7] * abs(w[7]), W[5, 8] * abs(w[8]), W[5, 4] * w[4], w[5]))
print("   d h5_9 / d x_8 = %.2f (comparators 7, 8 on at step 8), %.2f (8 clipped, 7 on), %.2f (unit 4 on at step 8); times U[8,5] = %.2f: %.1f, %.1f, %.1f logits per unit x_8"
      % (-(W[5, 7] * abs(w[7]) + W[5, 8] * abs(w[8])), -W[5, 7] * abs(w[7]), W[5, 4] * w[4], U[8, 5],
         -U[8, 5] * (W[5, 7] * abs(w[7]) + W[5, 8] * abs(w[8])), -U[8, 5] * W[5, 7] * abs(w[7]), U[8, 5] * W[5, 4] * w[4]))
print("   exact one-step sensitivities with every unit's gate open: W[5,ORD] w[ORD] = %.2f, W[9,ORD] w[ORD] = %.2f; U[8,5] * %.2f = %.1f, U[8,9] * %.2f = %.1f"
      % (W[5, ORD] @ w[ORD], W[9, ORD] @ w[ORD], W[5, ORD] @ w[ORD], U[8, 5] * (W[5, ORD] @ w[ORD]), W[9, ORD] @ w[ORD], U[8, 9] * (W[9, ORD] @ w[ORD])))

# numerical identity
x = sample(200_000, 1043)
H = run(x); L = logits(H)
b = H[:, 8] @ W.T
th = {j: b[:, j] / abs(w[j]) for j in COMP}; th[4] = -b[:, 4] / w[4]
x9 = x[:, 9]
Lf = sum(np.outer(np.maximum(th[j] - x9, 0), U[:, j] * abs(w[j])) for j in COMP)
Lf += np.outer(np.maximum(x9 - th[4], 0), U[:, 4] * w[4])
Lf += np.maximum(b[:, REST] + np.outer(x9, w[REST]), 0) @ U[:, REST].T
print(f"identity check on n = {len(x)} (seed 1043): max |formula - logits| = {np.abs(Lf - L).max():.2e}")

# ------------------------------------------------------------------------------------------------ Part B
print("\n=== Part B: sweep of the last input x_9 (seed 1040, n = 20000 prefixes; S8 > 0 only) ===")
n = 20000
x = sample(n, 1040)
H = run(x); r = np.arange(n)
b = H[:, 8] @ W.T
srt = np.argsort(-x[:, :9], axis=1)
p1o, p2o, p3o = srt[:, 0], srt[:, 1], srt[:, 2]
M8, S8, T8 = x[r, p1o], x[r, p2o], x[r, p3o]
ok = S8 > 0
grid = np.linspace(-0.6, 1.8, 1201)             # g = (x_9 - S8)/(M8 - S8)
ans = np.zeros((n, len(grid)), np.int8); D1 = np.zeros((n, len(grid))); D2 = np.zeros((n, len(grid)))
comp_lo = np.zeros((n, len(grid)), np.int8)
for k, g in enumerate(grid):
    xg = S8 + g * (M8 - S8)
    Lg = np.maximum(b + xg[:, None] * w, 0) @ U.T
    ans[:, k] = Lg.argmax(1)
    D1[:, k] = Lg[:, 9] - Lg[r, p1o]; D2[:, k] = Lg[:, 9] - Lg[r, p2o]
    comp_lo[:, k] = Lg[:, :9].argmax(1)
is9 = ans == 9
has = is9.any(1)
klo = is9.argmax(1); khi = len(grid) - 1 - is9[:, ::-1].argmax(1)
sel = ok & has
print(f"prefixes with S8 > 0: {ok.sum()}; of these the network answers 9 for some x_9 in the grid: {sel.sum() / ok.sum():.4f}")
sw = (np.diff(is9.astype(int), axis=1) != 0).sum(1)
print(f"the answer-9 set is a single interval (exactly 2 switches) in {np.mean(sw[sel] == 2):.4f} of them")
glo, ghi = grid[klo], grid[khi]
print("lower edge g_lo (ideal 0): percentiles 5/25/50/75/95 =", np.percentile(glo[sel], [5, 25, 50, 75, 95]).round(3))
print("upper edge g_hi (ideal 1): percentiles 5/25/50/75/95 =", np.percentile(ghi[sel], [5, 25, 50, 75, 95]).round(3))
xlo = S8 + glo * (M8 - S8); xhi = S8 + ghi * (M8 - S8)
for nm, y, t in (("lower edge x_lo vs S8", xlo, S8), ("upper edge x_hi vs M8", xhi, M8)):
    X = np.stack([t[sel], np.ones(sel.sum())], 1); beta, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
    res = y[sel] - X @ beta
    print(f"  [FIT] {nm}: slope {beta[0]:.3f}, intercept {beta[1]:+.3f}, R^2 = {1 - res.var() / y[sel].var():.4f}, residual sd {res.std():.3f}; "
          f"median |edge - target| = {np.median(np.abs(y[sel] - t[sel])):.3f}")
# competitor just below the lower edge and just above the upper edge
cl = comp_lo[r, np.maximum(klo - 1, 0)]; ch = comp_lo[r, np.minimum(khi + 1, len(grid) - 1)]
print(f"competitor (best stored position) just below the lower edge: old runner-up in {np.mean(cl[sel] == p2o[sel]):.3f}, old max in {np.mean(cl[sel] == p1o[sel]):.3f}, old third in {np.mean(cl[sel] == p3o[sel]):.3f}")
print(f"competitor just above the upper edge: old max in {np.mean(ch[sel] == p1o[sel]):.3f}, old runner-up in {np.mean(ch[sel] == p2o[sel]):.3f}")
print("P(answer 9), and medians of L9 - L[old max], L9 - L[old runner-up], along the sweep:")
for g0 in (-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5):
    k = np.abs(grid - g0).argmin()
    print(f"  g = {grid[k]:5.2f}: P(9) = {is9[ok, k].mean():.3f}   L9-L[p1old] {np.median(D1[ok, k]):7.2f}   L9-L[p2old] {np.median(D2[ok, k]):7.2f}")
dg = grid[1] - grid[0]
sl1 = np.gradient(D1, axis=1) / dg / (M8 - S8)[:, None]; sl2 = np.gradient(D2, axis=1) / dg / (M8 - S8)[:, None]
print("slopes per unit of x_9 (median, IQR):")
for g0 in (-0.5, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.7):
    k = np.abs(grid - g0).argmin()
    print(f"  g = {grid[k]:5.2f}: d(L9-L[p1old])/dx9 {np.median(sl1[ok, k]):7.1f} ({np.percentile(sl1[ok, k], 25):7.1f}..{np.percentile(sl1[ok, k], 75):7.1f})   "
          f"d(L9-L[p2old])/dx9 {np.median(sl2[ok, k]):7.1f} ({np.percentile(sl2[ok, k], 25):7.1f}..{np.percentile(sl2[ok, k], 75):7.1f})")
th = {j: b[:, j] / abs(w[j]) for j in COMP}; th[4] = -b[:, 4] / w[4]
print("where the thresholds at step 9 sit, in units g = (theta - S8)/(M8 - S8): percentiles 5/25/50/75/95")
for j in COMP + [4]:
    print(f"  theta_{j}: ", np.percentile(((th[j] - S8) / (M8 - S8))[ok], [5, 25, 50, 75, 95]).round(3))
# the stored comparison value: extrapolate the low-regime line of L9 - L[p2old] to zero
xref = np.minimum.reduce([th[j] for j in COMP + [4]]) - 0.5
Lr = np.maximum(b + xref[:, None] * w, 0) @ U.T
slope = slope_low[9] - slope_low[p2o]
xstar = xref - (Lr[:, 9] - Lr[r, p2o]) / slope
X = np.stack([S8[ok], np.ones(ok.sum())], 1); beta, *_ = np.linalg.lstsq(X, xstar[ok], rcond=None); res = xstar[ok] - X @ beta
print(f"[FIT] zero of the low-regime line of L9 - L[p2old] (all gates as at x_ref = min theta - 0.5) vs S8: slope {beta[0]:.3f}, intercept {beta[1]:+.3f}, "
      f"R^2 = {1 - res.var() / xstar[ok].var():.4f}; median |x* - S8| = {np.median(np.abs(xstar[ok] - S8[ok])):.3f}")

# ------------------------------------------------------------------------------------------------ Part C
print("\n=== Part C: sweep of x_8 with the other nine inputs fixed (seed 1041, n = 20000) ===")
x = sample(n, 1041)
H = run(x)
oth = np.delete(x, 8, axis=1); pos_oth = np.delete(np.arange(10), 8)
so = np.argsort(-oth, axis=1)
q1, q2 = pos_oth[so[:, 0]], pos_oth[so[:, 1]]          # positions of max / runner-up among the other nine
Mo, So = x[r, q1], x[r, q2]
ok = So > 0
h7 = H[:, 7]
ans = np.zeros((n, len(grid)), np.int8); E1 = np.zeros((n, len(grid))); E2 = np.zeros((n, len(grid))); h5 = np.zeros((n, len(grid))); h9u = np.zeros((n, len(grid)))
for k, g in enumerate(grid):
    xg = So + g * (Mo - So)
    h8 = np.maximum(h7 @ W.T + xg[:, None] * w, 0)
    h9 = np.maximum(h8 @ W.T + x[:, 9:10] * w, 0)
    Lg = h9 @ U.T
    ans[:, k] = Lg.argmax(1); E1[:, k] = Lg[:, 8] - Lg[r, q1]; E2[:, k] = Lg[:, 8] - Lg[r, q2]
    h5[:, k] = h9[:, 5]; h9u[:, k] = h9[:, 9]
is8 = ans == 8
has = is8.any(1); sel = ok & has
klo = is8.argmax(1); khi = len(grid) - 1 - is8[:, ::-1].argmax(1)
print(f"prefixes with S' > 0: {ok.sum()}; network answers 8 for some x_8: {sel.sum() / ok.sum():.4f}")
sw = (np.diff(is8.astype(int), axis=1) != 0).sum(1)
print(f"the answer-8 set is a single interval in {np.mean(sw[sel] == 2):.4f} of them")
glo, ghi = grid[klo], grid[khi]
print("lower edge g_lo (ideal 0): percentiles 5/25/50/75/95 =", np.percentile(glo[sel], [5, 25, 50, 75, 95]).round(3))
print("upper edge g_hi (ideal 1): percentiles 5/25/50/75/95 =", np.percentile(ghi[sel], [5, 25, 50, 75, 95]).round(3))
for case, cm in (("x_9 is neither max nor runner-up of the others", (q1 != 9) & (q2 != 9)), ("x_9 is the max of the others", q1 == 9), ("x_9 is the runner-up of the others", q2 == 9)):
    s2 = sel & cm
    print(f"  {case:48s} n = {s2.sum():5d}: g_lo median {np.median(glo[s2]):6.3f} (IQR {np.percentile(glo[s2], 25):6.3f}..{np.percentile(glo[s2], 75):6.3f}),"
          f" g_hi median {np.median(ghi[s2]):6.3f} (IQR {np.percentile(ghi[s2], 25):6.3f}..{np.percentile(ghi[s2], 75):6.3f})")
print("P(answer 8), medians of L8 - L[max of others], L8 - L[runner-up of others], h5_9 and h9_9 along the sweep:")
for g0 in (-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5):
    k = np.abs(grid - g0).argmin()
    print(f"  g = {grid[k]:5.2f}: P(8) = {is8[ok, k].mean():.3f}   L8-L[q1] {np.median(E1[ok, k]):7.2f}   L8-L[q2] {np.median(E2[ok, k]):7.2f}   h5_9 median {np.median(h5[ok, k]):6.2f}  P(h5_9>0) {np.mean(h5[ok, k] > 0):.3f}   h9_9 median {np.median(h9u[ok, k]):5.2f}  P(h9_9>0) {np.mean(h9u[ok, k] > 0):.3f}")
sl1 = np.gradient(E1, axis=1) / dg / (Mo - So)[:, None]; sl2 = np.gradient(E2, axis=1) / dg / (Mo - So)[:, None]; sl5 = np.gradient(h5, axis=1) / dg / (Mo - So)[:, None]
print("slopes per unit of x_8 (median, IQR):")
for g0 in (-0.5, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.7):
    k = np.abs(grid - g0).argmin()
    print(f"  g = {grid[k]:5.2f}: d(L8-L[q1])/dx8 {np.median(sl1[ok, k]):7.1f} ({np.percentile(sl1[ok, k], 25):7.1f}..{np.percentile(sl1[ok, k], 75):7.1f})   "
          f"d(L8-L[q2])/dx8 {np.median(sl2[ok, k]):7.1f} ({np.percentile(sl2[ok, k], 25):7.1f}..{np.percentile(sl2[ok, k], 75):7.1f})   d h5_9/dx8 {np.median(sl5[ok, k]):6.1f} ({np.percentile(sl5[ok, k], 25):6.1f}..{np.percentile(sl5[ok, k], 75):6.1f})")

# ------------------------------------------------------------------------------------------------ Part D
print("\n=== Part D [INTERVENTION]: zero one readout weight (seed 1042, n = 400000) ===")
x = sample(400_000, 1042)
H = run(x); h9 = H[:, 9]
order = np.argsort(-x, axis=1); p1, p2 = order[:, 0], order[:, 1]
cases = {"all": np.ones(len(x), bool), "max is last (p1=9)": p1 == 9, "runner-up is last (p2=9)": p2 == 9, "max at 8 (p1=8)": p1 == 8,
         "runner-up at 8 (p2=8)": p2 == 8, "neither in {8,9}": (p1 < 8) & (p2 < 8)}
print("intervention            " + "".join(f"{c:>26s}" for c in cases))
for nm, edits in (("none", []), ("U[9,4] = 0", [(9, 4)]), ("U[9,7] = 0", [(9, 7)]), ("U[8,5] = 0", [(8, 5)]), ("U[8,9] = 0", [(8, 9)]),
                  ("U[9,4] = +0.5 (as others)", [(9, 4, 0.5)]), ("U[8,5] = -0.4 (as others)", [(8, 5, -0.4)])):
    U2 = U.copy()
    for e in edits:
        U2[e[0], e[1]] = e[2] if len(e) == 3 else 0.0
    a = (h9 @ U2.T).argmax(1)
    print(f"{nm:24s}" + "".join(f"{np.mean(a[m] == p2[m]):26.4f}" for m in cases.values()))
print("what the network answers when the max is last and U[9,4] = 0:")
U2 = U.copy(); U2[9, 4] = 0; a = (h9 @ U2.T).argmax(1); m = p1 == 9
print(f"   answers 9 (the max): {np.mean(a[m] == 9):.4f}; answers the runner-up: {np.mean(a[m] == p2[m]):.4f}")
U2 = U.copy(); U2[8, 5] = 0; a = (h9 @ U2.T).argmax(1); m = p2 == 8
print(f"when the runner-up is at 8 and U[8,5] = 0: answers 8: {np.mean(a[m] == 8):.4f}; answers the max: {np.mean(a[m] == p1[m]):.4f}; answers the third: {np.mean(a[m] == order[m, 2]):.4f}")
