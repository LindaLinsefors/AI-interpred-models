"""R4 / r04: section 6 (last-step formula for the newest logit, slopes, the interval of x_9 on which
the network answers 9, unit 5 as the route for position 8, readout-weight ablations).
Seeds 7051-7054.  Output: r04_laststep_out.txt
"""
from rlib import *

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


P("=== SECTION 6, from the weights")
cj = U[9, COMP] * G
P("coefficients of [theta_j - x_9]_+ in L_9 for units 1, 6, 7, 8: %s ; of [x_9 - theta_4]_+: %.3f" % (np.round(cj, 3), U[9, 4] * W4))
P("U[9,4] = %.3f, U[9,7] = %.3f, U[8,5] = %.3f, U[8,9] = %.3f" % (U[9, 4], U[9, 7], U[8, 5], U[8, 9]))
below = np.array([(U[9, ORD] - U[p, ORD]) @ w[ORD] for p in range(9)])
above = np.array([(U[9, WAVE] - U[p, WAVE]) @ w[WAVE] + (U[9, 4] - U[p, 4]) * W4 for p in range(9)])
P("d(L_9 - L_p)/dx_9, p = 0..8, all 13 ordinary units on, pulses off:", np.round(below, 2), " range %.1f..%.1f" % (below.min(), below.max()))
P("d(L_9 - L_p)/dx_9, p = 0..8, comparators cut, unit 4 on, wave units on:", np.round(above, 2), " range %.1f..%.1f" % (above.min(), above.max()))
P("extra slope if unit 5 / unit 9 is on at step 9 (p = 0..8): unit 5 %s ; unit 9 %s" % (
    np.round([(U[9, 5] - U[p, 5]) * w[5] for p in range(9)], 2), np.round([(U[9, 9] - U[p, 9]) * w[9] for p in range(9)], 2)))
c5 = W[5, ORD] @ w[ORD]
P("dh_9[5]/dx_8 with all 13 ordinary units on at step 8: %.3f (x U[8,5] = %.2f) ; comparators 7 and 8 alone: %.3f (x U[8,5] = %.2f)" % (
    c5, c5 * U[8, 5], W[5, 7] * w[7] + W[5, 8] * w[8], (W[5, 7] * w[7] + W[5, 8] * w[8]) * U[8, 5]))
P("   + W[5,5] w_5 = %.3f if unit 5 is on at step 8 ; + W[5,9] w_9 = %.3f if unit 9 is on at step 8" % (W[5, 5] * w[5], W[5, 9] * w[9]))

# ---------------------------------------------------------------- identity on data
x = gauss(200_000, 7051)
H, Z = fwd(x)
pre = H[:, 8] @ W.T
theta = pre[:, COMP] / G
theta4 = -pre[:, 4] / W4
x9 = x[:, 9]
rho9 = H[:, 9][:, WAVE] @ U[9, WAVE] + H[:, 9, 5] * U[9, 5] + H[:, 9, 9] * U[9, 9]
L9 = (np.maximum(theta - x9[:, None], 0) * cj).sum(1) + U[9, 4] * W4 * np.maximum(x9 - theta4, 0) + rho9
P("\nidentity L_9 = formula + rho_9 on 200k sequences (seed 7051): max abs error %.2e (max |L_9| %.0f)" % (np.abs(L9 - H[:, 9] @ U[9]).max(), np.abs(L9).max()))
# measured slopes by finite difference
d = 1e-6
xp = x.copy(); xp[:, 9] += d
Hp, Zp = fwd(xp)
dl = (Hp[:, 9] - H[:, 9]) @ U.T / d
sl = dl[:, 9:10] - dl[:, :9]
same = ((Zp[:, 9] > 0) == (Z[:, 9] > 0)).all(1)
lo = same & (Z[:, 9][:, COMP] > 0).all(1) & (Z[:, 9, 4] <= 0)
hi = same & (Z[:, 9][:, COMP] <= 0).all(1) & (Z[:, 9, 4] > 0)
for name, m in (("x_9 below all five thresholds", lo), ("x_9 above all five thresholds", hi)):
    q = np.percentile(sl[m], [0, 5, 25, 50, 75, 95, 100])
    P("measured d(L_9 - L_p)/dx_9, %s (share %.3f): min %.1f, 5%% %.1f, quartiles %.1f / %.1f / %.1f, 95%% %.1f, max %.1f" % (name, m.mean(), *q))
    P("    share of (sequence, p) pairs inside the report's range: %.3f" % (((sl[m] >= (61.0 if m is lo else -57.8) - 0.05) & (sl[m] <= (62.1 if m is lo else -54.6) + 0.05)).mean()))
allon = lo & (Z[:, 9][:, WAVE] > 0).all(1) & (Z[:, 9, 5] <= 0) & (Z[:, 9, 9] <= 0)
P("    below + all wave units on + units 5, 9 off (share %.3f): range %.2f..%.2f" % (allon.mean(), sl[allon].min(), sl[allon].max()))

# ---------------------------------------------------------------- sweep of x_9
P("\nsweep of x_9 for fixed prefixes (seed 7052): grid -4..7 step 0.004")
xs = gauss(26_000, 7052)[:, :9]
st = prefix_stats(xs)
M8, S8, pM8, pS8 = st["M"][:, -1], st["S"][:, -1], st["pM"][:, -1], st["pS"][:, -1]
keep = S8 > 0
xs, M8, S8, pM8, pS8 = xs[keep][:20000], M8[keep][:20000], S8[keep][:20000], pM8[keep][:20000], pS8[keep][:20000]
h8 = fwd(xs)[0][:, -1]
grid = np.arange(-4.0, 7.0001, 0.004)
nint = np.zeros(len(xs), int); lo_e = np.full(len(xs), np.nan); hi_e = np.full(len(xs), np.nan)
below_ans = np.full(len(xs), -1); above_ans = np.full(len(xs), -1)
CH = 200
for i in range(0, len(xs), CH):
    b = h8[i:i + CH] @ W.T                                         # (c, 16)
    h9 = np.maximum(grid[None, :, None] * w + b[:, None, :], 0.0)   # (c, G, 16)
    a = (h9 @ U.T).argmax(-1)                                      # (c, G)
    is9 = a == 9
    rise = (is9[:, 1:] & ~is9[:, :-1]).sum(1) + is9[:, 0]
    nint[i:i + CH] = rise
    for r in range(len(b)):
        idx = np.nonzero(is9[r])[0]
        if len(idx):
            lo_e[i + r], hi_e[i + r] = grid[idx[0]], grid[idx[-1]]
            if idx[0] > 0:
                below_ans[i + r] = a[r, idx[0] - 1]
            if idx[-1] < len(grid) - 1:
                above_ans[i + r] = a[r, idx[-1] + 1]
n = len(xs)
some = nint >= 1
single = nint == 1
inside = single & (lo_e > grid[0]) & (hi_e < grid[-1])
P("prefixes with S_8 > 0: %d ; answers 9 somewhere: %.4f ; of those: exactly one interval %.4f, two %.4f, three or more %.4f" % (
    n, some.mean(), single.sum() / some.sum(), (nint == 2).sum() / some.sum(), (nint >= 3).sum() / some.sum()))
P("   single interval with both edges inside the grid: %.4f of those that answer 9 somewhere" % (inside.sum() / some.sum()))
for name, e, ref in (("lower edge vs S_8", lo_e, S8), ("upper edge vs M_8", hi_e, M8)):
    X = np.c_[ref[inside], np.ones(inside.sum())]
    c, *_ = np.linalg.lstsq(X, e[inside], rcond=None)
    r = e[inside] - X @ c
    P("   %s: slope %.3f, intercept %+.3f, R2 %.4f, median abs error %.3f" % (name, c[0], c[1], 1 - r.var() / e[inside].var(), np.median(np.abs(e[inside] - ref[inside]))))
gl = (lo_e - S8) / (M8 - S8); gh = (hi_e - S8) / (M8 - S8)
P("   edges in units g = (x_9 - S_8)/(M_8 - S_8): lower median %.3f quartiles %.3f / %.3f, 5%% %.2f ; upper median %.3f quartiles %.3f / %.3f, 95%% %.2f" % (
    np.median(gl[inside]), *np.percentile(gl[inside], [25, 75]), np.percentile(gl[inside], 5), np.median(gh[inside]), *np.percentile(gh[inside], [25, 75]), np.percentile(gh[inside], 95)))
P("   answer just below the lower edge is the old runner-up's position: %.3f ; just above the upper edge is the old max's position: %.3f" % (
    (below_ans[inside] == pS8[inside]).mean(), (above_ans[inside] == pM8[inside]).mean()))
# all prefixes (no S_8 > 0 filter) for comparison
xa = gauss(8000, 7054)[:, :9]
h8a = fwd(xa)[0][:, -1]
na = np.zeros(len(xa), int)
for i in range(0, len(xa), CH):
    is9 = (np.maximum(grid[None, :, None] * w + (h8a[i:i + CH] @ W.T)[:, None, :], 0.0) @ U.T).argmax(-1) == 9
    na[i:i + CH] = (is9[:, 1:] & ~is9[:, :-1]).sum(1) + is9[:, 0]
P("   all prefixes (8000, no filter): answers 9 somewhere %.4f ; one interval %.4f of those" % ((na >= 1).mean(), (na == 1).sum() / (na >= 1).sum()))

# ---------------------------------------------------------------- unit 5 as the route for position 8
x = gauss(200_000, 7053)
H, Z = fwd(x)
xp = x.copy(); xp[:, 8] += d
Hp, Zp = fwd(xp)
same = ((Zp[:, 8:] > 0) == (Z[:, 8:] > 0)).all((1, 2))
s5 = (Hp[:, 9, 5] - H[:, 9, 5]) / d
on5 = Z[:, 9, 5] > 0
c78on = (Z[:, 8, 7] > 0) & (Z[:, 8, 8] > 0) & (Z[:, 8, 4] <= 0)
P("\ndh_9[5]/dx_8 by finite difference (seed 7053), unit 5 on at step 9, gates unchanged:")
for name, m in (("comparators 7, 8 on and unit 4 off at step 8", same & on5 & c78on),
                ("  of which all 13 ordinary on and pulses off at step 8", same & on5 & c78on & (Z[:, 8][:, ORD] > 0).all(1) & (Z[:, 8, 5] <= 0) & (Z[:, 8, 9] <= 0)),
                ("comparator 8 off, 7 on, unit 4 off", same & on5 & (Z[:, 8, 7] > 0) & (Z[:, 8, 8] <= 0) & (Z[:, 8, 4] <= 0)),
                ("all comparators off and unit 4 on (new max at 8)", same & on5 & (Z[:, 8][:, COMP] <= 0).all(1) & (Z[:, 8, 4] > 0))):
    q = np.percentile(s5[m], [5, 25, 50, 75, 95])
    P("   %-58s n=%6d  5%% %.2f quartiles %.2f / %.2f / %.2f 95%% %.2f" % (name, m.sum(), *q))

# ---------------------------------------------------------------- readout ablations
x = gauss(400_000, 7054)
H, _ = fwd(x)
st = prefix_stats(x)
pM, pS = st["pM"][:, -1], st["pS"][:, -1]
P("\nreadout-weight ablations (n = 400k, seed 7054): accuracy all / max last / runner-up last / runner-up at 8")
for name, (i, j, v) in (("none", (0, 0, U[0, 0])), ("U[8,5] = 0", (8, 5, 0.0)), ("U[9,7] = 0", (9, 7, 0.0)), ("U[9,4] = 0", (9, 4, 0.0)), ("U[8,9] = 0", (8, 9, 0.0))):
    U2 = U.copy(); U2[i, j] = v
    ok = (H[:, -1] @ U2.T).argmax(1) == pS
    P("   %-11s %.4f / %.4f / %.4f / %.4f" % (name, ok.mean(), ok[pM == 9].mean(), ok[pS == 9].mean(), ok[pS == 8].mean()))
out.close()
