"""E30. The lead's own recomputation of the numbers on which the figure checks of 2026-11-20 disagreed with
report.md (plots/index.md lists them). Each block prints the value and the report's old wording.
Run:  python -B hide/e30_after_figures.py > hide/outputs/e30_after_figures.txt      (about 2 min)"""
from itertools import groupby

from common import *
from modes import LEFT, LAM, coords

w4, g = w[4], -w
ar = lambda n: np.arange(n)
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
med = lambda v: float(np.median(v)) if len(v) else float("nan")


def head(s):
    print("\n" + "=" * 8, s)


# ---------------------------------------------------------------- A. update-rule coefficients (section 5)
head("A. comparator terms of the update rule, l[k] g_k / w4   [report: comparator 7 off: -0.06, -5.39]")
for name in ("level", "runner", "clock"):
    c = np.array([LEFT[name][ORD.index(k)] * g[k] / w4 for k in COMP])
    if name == "clock":
        print(f"  {name:7s}", "  ".join(f"k={k}: {abs(v):.4f} at {np.degrees(np.angle(v)):7.2f}" for k, v in zip(COMP, c)),
              f"  sum {abs(c.sum()):.4f}")
    else:
        print(f"  {name:7s}", "  ".join(f"k={k}: {v.real:+.4f}" for k, v in zip(COMP, c)), f"  sum {c.sum().real:+.4f}")

# ---------------------------------------------------------------- K. readout columns: frequency free
head("K. readout columns U[0..7, j]: fit a cos(om i) + b sin(om i) + c   [report: clock frequency 42.75, R2 .974 .943 .991 .988]")
i8 = np.arange(8)
def r2_at(col, om):
    X = np.stack([np.cos(om * i8), np.sin(om * i8), np.ones(8)], 1)
    res = col - X @ np.linalg.lstsq(X, col, rcond=None)[0]
    return 1 - res @ res / ((col - col.mean()) ** 2).sum()
grid = np.radians(np.arange(30, 60.001, 0.01))
for j in COMP:
    r = np.array([r2_at(U[:8, j], om) for om in grid])
    print(f"  unit {j}: R2 at 42.75 = {r2_at(U[:8, j], np.radians(42.75)):.3f};  best frequency {np.degrees(grid[r.argmax()]):.2f} deg per index, R2 = {r.max():.3f}")

# ---------------------------------------------------------------- L. spectrum of Sigma
head("L. eigenvalues of Sigma = U_o A pinv(U_o) against those of A   [report: level, runner, clock absent]")
A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
eS, eA = np.linalg.eigvals(Uo @ A @ np.linalg.pinv(Uo)), np.linalg.eigvals(A)
for lam in sorted(eA[eA.imag >= -1e-12], key=lambda v: -abs(v)):
    d = np.abs(eS - lam).min()
    print(f"  A: {abs(lam):.4f} at {np.degrees(np.angle(lam)):7.2f}   nearest eigenvalue of Sigma is {d:.3f} away")
print("  Sigma:", "  ".join(f"{abs(v):.3f}@{np.degrees(np.angle(v)):.1f}" for v in sorted(eS[eS.imag >= -1e-12], key=lambda v: -abs(v))))

# ---------------------------------------------------------------- data
B = 1_000_000
x = sample(B, 3030)
H, Z = run(x, return_pre=True)
st = running_stats(x)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Sp = np.maximum(S, 0)
cz = coords(H)
z, lev = cz["clock"] / w4, cz["level"] / w4
truth, ans = second_argmax(x), (H[:, 9] @ U.T).argmax(1)
print(f"\n{B} N(0,1) sequences, seed 3030; accuracy {np.mean(truth == ans):.4f}")

# ---------------------------------------------------------------- B. margin by distance (section 3.4)
head("B. median margin (s[pS] - s[pM]) / M at t = 9 by distance D   [report, gap 0.3-0.5: 4.8 7.8 10.2 13.7 16.8 16.4 17.9 21.9]")
s9 = H[:, 9] @ U.T
V9 = H[:, 9][:, COMP] @ U[:, COMP].T
M9, S9, pM, pS = M[:, 9], S[:, 9], st["p1"][:, 9], st["p2"][:, 9]
ok = (H[:, 9][:, COMP] > 0).all(1) & (S9 > 0.2) & (pM < 9) & (pS < 9)
D, rel = np.abs(pM - pS), (M9 - S9) / M9
marg = (s9[ar(B), pS] - s9[ar(B), pM]) / M9
vpart = (V9[ar(B), pS] - V9[ar(B), pM]) / M9
has8 = (pM == 8) | (pS == 8)
for lo, hi in ((0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.80)):
    b = ok & (rel > lo) & (rel <= hi)
    print(f"  gap {lo:.2f}-{hi:.2f}  all pairs      :", " ".join(f"{med(marg[b & (D == d)]):6.1f}" for d in range(1, 9)))
    print(f"                 without pos. 8 :", " ".join(f"{med(marg[b & (D == d) & ~has8]):6.1f}" for d in range(1, 8)))
    print(f"                 V part, without:", " ".join(f"{med(vpart[b & (D == d) & ~has8]):6.1f}" for d in range(1, 8)))
print("  share of pairs that include position 8, D = 1..8:", " ".join(f"{has8[ok & (D == d)].mean():.2f}" for d in range(1, 9)))

# ---------------------------------------------------------------- C. clock laws (section 4.1)
head("C. alpha(a): mean over the runner-up's ages of |c1|, z ~ c1 (M - S) + c2 S on S > 0.05   [report: ... 0.85, 0.82, 0.81]")
okc = S > 0.05
def fit(sel):
    X = np.stack([(M - S)[sel], S[sel]], 1)
    return np.linalg.solve(X.T @ X, X.T @ z[sel])[0]
phi, alpha = [], []
for a in range(1, 10):
    phi.append(np.angle(fit(okc & (a1 == a))))
    alpha.append(np.mean([abs(fit(okc & (a1 == a) & (a2 == b))) for b in range(10) if b != a and (okc & (a1 == a) & (a2 == b)).sum() >= 2000]))
phi = np.array(phi)
print("  phi(a)  :", np.round(np.degrees(np.unwrap(phi)), 1))
print("  alpha(a):", np.round(alpha, 4))
big = (M > 0.3) & (a1 >= 1)
with np.errstate(divide="ignore", invalid="ignore"):
    gap, lag = (M - Sp) / M, np.degrees(wrap(phi[np.clip(a1, 1, 9) - 1] - np.angle(z)))      # > 0: z behind phi(a)
print("  signed median of phi(a) - arg z, and median |.|, by relative gap <0.1, 0.1-0.3, 0.3-0.6, >0.6   [report: 'it lags']")
for lo, hi in ((1, 1), (2, 4), (5, 9)):
    sel = big & (a1 >= lo) & (a1 <= hi)
    bins = [sel & (gap <= 0.1), sel & (gap > 0.1) & (gap <= 0.3), sel & (gap > 0.3) & (gap <= 0.6), sel & (gap > 0.6)]
    print(f"    ages {lo}-{hi}: signed", " ".join(f"{med(lag[b]):6.1f}" for b in bins), "   absolute", " ".join(f"{med(np.abs(lag[b])):6.1f}" for b in bins))
for a in (1, 4, 8):
    sel = big & (a1 == a) & (S / M > 0.95)
    print(f"  tie radius, age {a}: median |z| / M = {med((np.abs(z) / M)[sel]):.4f}  (n = {sel.sum()})   [report: 0.21 at age 1, 0.15 at age 8]")

# ---------------------------------------------------------------- D. level one step after a new maximum
head("D. level(t) / (w4 M_(t-1)) when M_(t-1) arrived at step t-1   [report: 0.99]")
r = lev[:, 1:] / M[:, :-1]
new = (a1[:, :-1] == 0) & (M[:, :-1] > 0)
print(f"  all such states: median {med(r[new]):.4f};  with a runner-up above 0.05: {med(r[new & (S[:, :-1] > 0.05)]):.4f};  first input (no runner-up): {med(r[:, 0][M[:, 0] > 0]):.4f}")
for lo, hi in ((2, 2), (3, 5), (6, 9)):
    sel = (a1[:, :-1] + 1 >= lo) & (a1[:, :-1] + 1 <= hi) & (M[:, :-1] > 0)
    print(f"  maximum {lo}-{hi} steps old at step t: median {med(r[sel]):.4f}")

# ---------------------------------------------------------------- E. wipe at t0 = 8 (section 3.5)
head("E. comparators set to zero after step 8   [report: rates 0.80 1.00 0.42 0.71 0.93 1.00 for k = 1..6; 'every miss answers position 8']")
pmax = x.argmax(1)
selw = (S[:, 9] > 0.2) & (M[:, 9] - S[:, 9] > 0.2)
h8 = H[:, 8].copy(); h8[:, COMP] = 0
aw = (np.maximum(x[:, 9, None] * w + h8 @ W.T, 0) @ U.T).argmax(1)
for k in range(1, 7):
    q = selw & (pmax == 8 - k)
    miss = q & (aw != pmax)
    print(f"  k = {k}: P(maximum) = {np.mean(aw[q] == pmax[q]):.3f};  of the {miss.sum()} other answers: position 8 in {np.mean(aw[miss] == 8):.3f}, "
          f"the true runner-up in {np.mean(aw[miss] == truth[miss]):.3f} (runner-up at position 8: {np.mean((aw[miss] == 8) & (truth[miss] == 8)):.3f})")
q = selw & (pmax >= 2) & (pmax <= 7); miss = q & (aw != pmax)
print(f"  k = 1..6 pooled: position 8 in {np.mean(aw[miss] == 8):.3f} of the other answers; runner-up (not at 8) in {np.mean((aw[miss] == truth[miss]) & (truth[miss] != 8)):.3f}")

# ---------------------------------------------------------------- I. units at the step of a new maximum (section 4.2)
head("I. unit value / old maximum at the step of a new maximum (t_r) and one step before, by age k of the old maximum at t_r")
tr = pmax
okn = tr >= 2
M0, p0 = M[ar(B), np.maximum(tr - 1, 0)], st["p1"][ar(B), np.maximum(tr - 1, 0)]
okn &= M0 > 0.3
for u, k in ((0, 2), (11, 5), (10, 6), (12, 3), (15, 7), (14, 8), (13, 4)):
    q = okn & (tr - p0 == k)
    now, before = H[ar(B), tr, u][q] / M0[q], H[ar(B), np.maximum(tr - 1, 0), u][q] / M0[q]
    print(f"  unit {u:2d}, old maximum {k} steps old: median {med(before):5.2f} M one step before, {med(now):5.2f} M at the new maximum "
          f"(exactly zero in {np.mean(now == 0):.2f});  U[{9 - k},{u}] = {U[9 - k, u]:+.2f}  (old maximum's index one step later: U[{8 - k},{u}] = {U[8 - k, u]:+.2f})")

# ---------------------------------------------------------------- J. resting gate pattern (section 4.3)
head("J. resting gate pattern (13 units on, units 4, 5, 9 off)   [report: 10% of states, never at ages 1-2, 2% at ages 5 and 6; at most four consecutive steps]")
G = Z > 0
rest = G[..., ORD].all(-1) & ~G[..., PULSE].any(-1)
print(f"  all states {rest.mean():.4f};  last step {rest[:, 9].mean():.4f}")
for label, cond in (("maximum > 0", M > 0), ("maximum > 0 and runner-up > 0.05", (M > 0) & (S > 0.05)), ("maximum > 0.3", M > 0.3)):
    print(f"  by age 0..9 of the maximum, {label}:", " ".join(f"{100 * rest[cond & (a1 == a)].mean():.2f}" for a in range(10)), "(%)")
runs = np.zeros(B, int); cur = np.zeros(B, int)
for t in range(T):
    cur = np.where(rest[:, t], cur + 1, 0); runs = np.maximum(runs, cur)
print("  longest run of consecutive resting steps, number of sequences:", {int(k): int(v) for k, v in zip(*np.unique(runs, return_counts=True))})

print("  number of resting states at ages 0, 1, 2 of a positive maximum:", [int(rest[(M > 0) & (a1 == a)].sum()) for a in range(3)],
      "of", [int(((M > 0) & (a1 == a)).sum()) for a in range(3)])
print(f"  sequences with at least 3, 4, 5 consecutive resting steps: {np.mean(runs >= 3):.4f}, {np.mean(runs >= 4):.5f}, {np.mean(runs >= 5):.6f}")

head("C'. signed median of phi(a) - arg z per age of the maximum (M > 0.3), relative gap 0.1-0.3 and 0.3-0.6   [report: 'lags by 8-15' and 'by 3-7']")
for lo, hi in ((0.1, 0.3), (0.3, 0.6)):
    print(f"  gap {lo}-{hi}, ages 1..9:", " ".join(f"{med(lag[big & (a1 == a) & (gap > lo) & (gap <= hi)]):5.1f}" for a in range(1, 10)),
          "  absolute:", " ".join(f"{med(np.abs(lag[big & (a1 == a) & (gap > lo) & (gap <= hi)])):5.1f}" for a in range(1, 10)))

# ---------------------------------------------------------------- G. third named instead of runner-up (section 7)
head("G. P(answer = third) in four bins of (S - T)/S: <=3%, 3-8%, 8-20%, 20-50% (third above 0.2)")
T3, pT = st["M3"][:, 9], st["p3"][:, 9]
rel3 = (S9 - T3) / S9
okg = T3 > 0.2
bins = [okg & (rel3 <= 0.03), okg & (rel3 > 0.03) & (rel3 <= 0.08), okg & (rel3 > 0.08) & (rel3 <= 0.20), okg & (rel3 > 0.20) & (rel3 <= 0.50)]
for label, cond in (("maximum arrives after both candidates", (pM > pS) & (pM > pT)), ("maximum at position 9", pM == 9),
                    ("maximum arrives before both candidates", (pM < pS) & (pM < pT)), ("maximum at position 0", pM == 0)):
    print(f"  {label:40s}:", " ".join(f"{np.mean(ans[b & cond] == pT[b & cond]):.3f}" for b in bins))
print("  single arrangements (runner-up, third, maximum), 400k rearranged sequences each   [report: (2,1,9), (6,5,9), (7,6,9): 0.23-0.25 in the third bin]")
xs = np.sort(sample(400_000, 3031), 1)                       # ascending: column 9 = maximum, 8 = runner-up, 7 = third
rest7 = np.random.default_rng(3032).permuted(xs[:, :7], axis=1)
for arr in ((0, 7, 9), (8, 0, 9), (7, 0, 9), (7, 8, 9), (2, 1, 9), (6, 5, 9), (7, 6, 9)):
    xa = np.empty_like(xs)
    xa[:, [k for k in range(10) if k not in arr]] = rest7
    xa[:, arr[0]], xa[:, arr[1]], xa[:, arr[2]] = xs[:, 8], xs[:, 7], xs[:, 9]
    aa = (run(xa)[:, 9] @ U.T).argmax(1)
    r3 = (xs[:, 8] - xs[:, 7]) / xs[:, 8]
    o = xs[:, 7] > 0.2
    bb = [o & (r3 <= 0.03), o & (r3 > 0.03) & (r3 <= 0.08), o & (r3 > 0.08) & (r3 <= 0.20), o & (r3 > 0.20) & (r3 <= 0.50)]
    print(f"    {arr}:", " ".join(f"{np.mean(aa[b] == arr[1]):.3f}" for b in bb), "  n =", [int(b.sum()) for b in bb])

# ---------------------------------------------------------------- more data for three last digits
head("M. six more samples of 1M sequences: alpha(9), tie radius at ages 1 and 8, runs of resting steps")
al9, tie1, tie8, runs5, nseq = [], [], [], 0, 0
for seed in range(3040, 3046):
    xb = sample(B, seed)
    Hb, Zb = run(xb, return_pre=True)
    sb = running_stats(xb)
    Mb, Sb, a1b, a2b = sb["M1"], sb["M2"], sb["a1"], sb["a2"]
    zb = coords(Hb)["clock"] / w4
    cells = []
    for b in range(9):
        sel = (Sb > 0.05) & (a1b == 9) & (a2b == b)
        X = np.stack([(Mb - Sb)[sel], Sb[sel]], 1)
        cells.append(abs(np.linalg.solve(X.T @ X, X.T @ zb[sel])[0]))
    al9.append(np.mean(cells))
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = (Mb > 0.3) & (Sb / Mb > 0.95)
    tie1.append((np.abs(zb) / Mb)[tb & (a1b == 1)]); tie8.append((np.abs(zb) / Mb)[tb & (a1b == 8)])
    Gb = Zb > 0
    rb = Gb[..., ORD].all(-1) & ~Gb[..., PULSE].any(-1)
    cur = np.zeros(B, int); best = np.zeros(B, int)
    for t in range(T):
        cur = np.where(rb[:, t], cur + 1, 0); best = np.maximum(best, cur)
    runs5 += int((best >= 5).sum()); nseq += B
    print(f"  seed {seed}: alpha(9) = {al9[-1]:.4f}, tie radius age 1 = {np.median(tie1[-1]):.4f}, age 8 = {np.median(tie8[-1]):.4f}, longest run {best.max()}")
print(f"  alpha(9): mean {np.mean(al9):.4f} +- {np.std(al9, ddof=1) / np.sqrt(len(al9)):.4f}")
print(f"  tie radius, pooled: age 1 {np.median(np.concatenate(tie1)):.4f}, age 8 {np.median(np.concatenate(tie8)):.4f} (n = {len(np.concatenate(tie8))})")
print(f"  sequences with five or more consecutive resting steps: {runs5} of {nseq}")
