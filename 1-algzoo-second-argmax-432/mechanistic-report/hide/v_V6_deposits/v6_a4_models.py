"""V6 attacks on D1/D2 using the natural-sample events of d1_events.npz (seeds 5000..5009, 4M sequences).

[1] event-level model comparison: how much of the variance of y does each description explain?
[2] pooled c1(Delta) against the cosine law and alternatives (the lead's own table, incl. Delta = 8).
[3] omitted variable: the excess of the new max, e = (x_t - M0) / M0.
[4] sign of the separation (which of the two is older), natural weights.
[5] D2 on a fixed cohort (composition control).
[6] sanity check of the index convention (score index 9 - age).
"""
from v6_data import *
from scipy.optimize import least_squares

out = open(HERE / "a4_models.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


e = dict(np.load(HERE / "d1_events.npz"))
y = e["yk"][:, 0]
gap, M0 = e["M0"] - e["S0"], e["M0"]
t, pM, pS = e["t"], e["pM"], e["pS"]
D = np.abs(pM - pS)
aM, aS = t - pM, t - pS                      # ages at the record step (aM >= 2 because g >= 1)
n = len(y)
sst = ((y - y.mean()) ** 2).sum()
P(f"{n} records (D1 selection, g >= 1), seeds 5000..5009")


def grouped(keys):
    """Free (c1, c2) per group; returns fitted values."""
    fit = np.zeros(n)
    _, inv = np.unique(keys, return_inverse=True)
    for k in range(inv.max() + 1):
        m = inv == k
        b = np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), y[m], rcond=None)[0]
        fit[m] = b[0] * gap[m] + b[1] * M0[m]
    return fit


def r2(fit):
    return 1 - ((y - fit) ** 2).sum() / sst


P("\n[1] centred R2 of y over ALL records (pooled, natural weights) for nested descriptions y ~ c1(.) (M0 - S0) + c2(.) M0")
b = np.linalg.lstsq(np.column_stack([gap, M0]), y, rcond=None)[0]
P(f"    {r2(b[0] * gap + b[1] * M0):.3f}   one c1, one c2 for everything (c1 = {b[0]:.1f}, c2 = {b[1]:.2f})")
f = 1 - np.cos(OMEGA * D)
b = np.linalg.lstsq(np.column_stack([f * gap, M0]), y, rcond=None)[0]
P(f"    {r2(b[0] * f * gap + b[1] * M0):.3f}   cosine law: c1 = K (1 - cos(42.75 Delta)), one c2 (K = {b[0]:.1f}, c2 = {b[1]:.2f})")
X = np.column_stack([f * gap] + [M0 * (D == d) for d in range(1, 9)])
b = np.linalg.lstsq(X, y, rcond=None)[0]
P(f"    {r2(X @ b):.3f}   cosine law with a free c2 per Delta (K = {b[0]:.1f})")
X = np.column_stack([D * gap, M0])
b = np.linalg.lstsq(X, y, rcond=None)[0]
P(f"    {r2(X @ b):.3f}   linear law: c1 = a Delta, one c2 (a = {b[0]:.1f})")
P(f"    {r2(grouped(D)):.3f}   free (c1, c2) per Delta  [the lead's D1 table; 8 groups]")
P(f"    {r2(grouped(pM - pS)):.3f}   free (c1, c2) per SIGNED separation pM0 - pS0 (15 groups)")
P(f"    {r2(grouped(aS)):.3f}   free (c1, c2) per age of the old runner-up (9 groups)")
P(f"    {r2(grouped(aM)):.3f}   free (c1, c2) per age of the old max (8 groups)")
P(f"    {r2(grouped(aM * 10 + aS)):.3f}   free (c1, c2) per pair of ages (aM, aS) (64 groups)")
P(f"    {r2(grouped(t * 100 + aM * 10 + aS)):.3f}   free (c1, c2) per configuration (t, pM0, pS0) (204 groups)")

P("\n[2] pooled c1(Delta) (natural weights) against one-parameter-family laws; relative error per Delta")
c1 = np.array([np.linalg.lstsq(np.column_stack([gap[D == d], M0[D == d]]), y[D == d], rcond=None)[0][0] for d in range(1, 9)])
dd = np.arange(1, 9)
P("    Delta:                         " + " ".join(f"{d:7d}" for d in dd))
P("    my pooled c1:                  " + " ".join(f"{v:7.1f}" for v in c1))
for nm, rng_ in (("fit on Delta 1..7", slice(0, 7)), ("fit on Delta 1..8", slice(0, 8))):
    P(f"    -- {nm} (unweighted least squares on the c1 values)")
    d_, c_ = dd[rng_], c1[rng_]
    tot = ((c_ - c_.mean()) ** 2).sum()
    fits = {}
    ff = 1 - np.cos(OMEGA * d_); K = ff @ c_ / (ff @ ff)
    fits[f"K(1-cos(42.75 D)), K={K:.1f}"] = lambda q, K=K: K * (1 - np.cos(OMEGA * q))
    r = least_squares(lambda p: p[0] * (1 - np.cos(np.radians(p[1]) * d_)) - c_, [45, 42.75])
    fits[f"K(1-cos(w D)), K={r.x[0]:.1f}, w={r.x[1]:.2f}"] = lambda q, p=r.x: p[0] * (1 - np.cos(np.radians(p[1]) * q))
    r = least_squares(lambda p: p[0] * (1 - 0.951 ** d_ * np.cos(OMEGA * d_)) - c_, [45])
    fits[f"K(1-0.951^D cos(42.75 D)), K={r.x[0]:.1f}"] = lambda q, p=r.x: p[0] * (1 - 0.951 ** q * np.cos(OMEGA * q))
    r = least_squares(lambda p: p[0] * (1 - p[1] ** d_ * np.cos(np.radians(p[2]) * d_)) - c_, [45, 0.95, 42.75])
    fits[f"K(1-r^D cos(w D)), K={r.x[0]:.1f}, r={r.x[1]:.3f}, w={r.x[2]:.2f}"] = lambda q, p=r.x: p[0] * (1 - p[1] ** q * np.cos(np.radians(p[2]) * q))
    a = d_ @ c_ / (d_ @ d_)
    fits[f"a D, a={a:.1f}"] = lambda q, a=a: a * q
    Xq = np.column_stack([d_, d_ ** 2]); bq = np.linalg.lstsq(Xq, c_, rcond=None)[0]
    fits[f"a D + b D^2, a={bq[0]:.1f}, b={bq[1]:.2f}"] = lambda q, bq=bq: bq[0] * q + bq[1] * q ** 2
    fits[f"constant {c_.mean():.1f}"] = lambda q, m=c_.mean(): m + 0 * q
    for k, fn in fits.items():
        pr = fn(dd.astype(float))
        P(f"       R2 = {1 - ((c_ - fn(d_.astype(float))) ** 2).sum() / tot:6.3f}  {k:48s} pred: " + " ".join(f"{v:6.1f}" for v in pr)
          + "  | obs/pred: " + " ".join(f"{o / p_:5.2f}" for o, p_ in zip(c1, pr)))

P("\n[3] omitted variable: excess of the new max e = (x_t - M0)/M0. c1 by Delta within terciles of e (natural weights)")
ex = (e["xt"] - M0) / M0
q = np.quantile(ex, [1 / 3, 2 / 3])
P(f"    tercile edges of e: {q[0]:.3f}, {q[1]:.3f}")
for lo, hi, nm in ((-np.inf, q[0], "small e"), (q[0], q[1], "middle e"), (q[1], np.inf, "large e")):
    row, rowp = [], []
    for d in range(1, 9):
        m = (D == d) & (ex > lo) & (ex <= hi)
        row.append(np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), y[m], rcond=None)[0][0]); rowp.append(np.mean(y[m] > 0))
    P(f"    {nm:9s}: c1 = " + " ".join(f"{v:6.1f}" for v in row) + "   P(y>0) = " + " ".join(f"{v:.3f}" for v in rowp))
m = ex < 0.05
row, rowp = [], []
for d in range(1, 9):
    mm = m & (D == d)
    row.append(np.linalg.lstsq(np.column_stack([gap[mm], M0[mm]]), y[mm], rcond=None)[0][0]); rowp.append(np.mean(y[mm] > 0))
P(f"    e < 0.05 : c1 = " + " ".join(f"{v:6.1f}" for v in row) + "   P(y>0) = " + " ".join(f"{v:.3f}" for v in rowp) + f"   (n = {m.sum()})")
X3 = np.column_stack([gap, M0, e["xt"] - M0])
for d in range(1, 9):
    m = D == d
    b2 = np.linalg.lstsq(X3[m, :2], y[m], rcond=None)[0]
    b3 = np.linalg.lstsq(X3[m], y[m], rcond=None)[0]
    r2a = 1 - ((y[m] - X3[m, :2] @ b2) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    r2b = 1 - ((y[m] - X3[m] @ b3) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    P(f"    Delta={d}: adding (x_t - M0) as a third regressor: coefficient {b3[2]:6.2f}, c1 {b2[0]:6.1f} -> {b3[0]:6.1f}, R2 {r2a:.3f} -> {r2b:.3f}")

P("\n[4] which is older? c1, c2, R2, P(y>0) by SIGNED separation pM0 - pS0 (negative: old runner-up YOUNGER than the old max)")
for sd_ in range(-8, 8):
    m = (pM - pS) == sd_
    if m.sum() < 500:
        continue
    b = np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), y[m], rcond=None)[0]
    r = y[m] - b[0] * gap[m] - b[1] * M0[m]
    P(f"    pM0 - pS0 = {sd_:+d}: n = {m.sum():7d}  c1 = {b[0]:6.1f}  c2 = {b[1]:6.2f}  R2 = {1 - (r ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum():.3f}  "
      f"P(y>0) = {np.mean(y[m] > 0):.3f}   c1/(1-cos) = {b[0] / (1 - np.cos(OMEGA * abs(sd_))):6.1f}")

P("\n[5] D2 with the composition held fixed: records at t <= 4 whose top two stay unchanged up to t + 5 (same records at every k)")
yk, ok2 = e["yk"], e["ok2"]
for nm, coh in (("g >= 1 (D1 records)", ok2[:, 5]),):
    row, row2, rowp = [], [], []
    for k in range(6):
        b = np.linalg.lstsq(np.column_stack([gap[coh], M0[coh]]), yk[coh, k], rcond=None)[0]
        row.append(b[0]); row2.append(b[1]); rowp.append(np.mean(yk[coh, k] > 0))
    P(f"    {nm}: n = {coh.sum()}, Delta in {sorted(set(D[coh]))}: c1 = " + ", ".join(f"{v:.1f}" for v in row) + " | c2 = "
      + ", ".join(f"{v:.2f}" for v in row2) + " | P(y>0) = " + ", ".join(f"{v:.3f}" for v in rowp))
P("    per Delta within that cohort:")
for d in (1, 2, 3):
    coh = ok2[:, 5] & (D == d)
    if coh.sum() < 300:
        continue
    row = [np.linalg.lstsq(np.column_stack([gap[coh], M0[coh]]), yk[coh, k], rcond=None)[0][0] for k in range(6)]
    P(f"      Delta={d} (n = {coh.sum()}): c1 = " + ", ".join(f"{v:.1f}" for v in row))
P("    cohorts by last valid k (records with top two unchanged up to t + K, t <= 9 - K), c1 at k = 0..K:")
for K in range(1, 6):
    coh = ok2[:, K]
    row = [np.linalg.lstsq(np.column_stack([gap[coh], M0[coh]]), yk[coh, k], rcond=None)[0][0] for k in range(K + 1)]
    P(f"      K={K} (n = {coh.sum()}): " + ", ".join(f"{v:.1f}" for v in row))
P("    per (Delta, k), top two unchanged (all records valid at k, natural weights):")
for d in range(1, 8):
    row = []
    for k in range(6):
        m = ok2[:, k] & (D == d)
        row.append(f"{np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), yk[m, k], rcond=None)[0][0]:6.1f}" if m.sum() > 300 else "     -")
    P(f"      Delta={d}: " + " ".join(row))

P("\n[6] index convention check (seed 5020, 200k): P(argmax_{i >= 9-t} s_t[i] + t - 9 == second argmax of x_0..x_t | S_t > 0)")
x = sample(200_000, 5020)
H = run(x)
vals, pos = top3(x)
row = []
for tt in range(1, 10):
    s = H[:, tt] @ U.T
    guess = s[:, 9 - tt:].argmax(1)          # index within the slice = position
    m = vals[:, tt, 1] > 0
    row.append(np.mean(guess[m] == pos[m, tt, 1]))
P("    t = 1..9: " + " ".join(f"{v:.3f}" for v in row))
row = []
for tt in range(1, 10):
    s = H[:, tt] @ U.T
    # deliberately shifted convention (index 9 - a - 1 for age a) as a negative control
    guess = s[:, 8 - tt: 9].argmax(1) if tt < 9 else s[:, 0:9].argmax(1) + 1
    m = vals[:, tt, 1] > 0
    row.append(np.mean(guess[m] == pos[m, tt, 1]))
P("    shifted-by-one convention (negative control): " + " ".join(f"{v:.3f}" for v in row))
out.close()
