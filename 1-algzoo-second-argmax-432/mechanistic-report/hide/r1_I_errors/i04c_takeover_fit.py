"""I4 (detail). What sets the takeover offset eps* of i04b_takeover.py?  Uses the saved sweeps.
RT: eps* against the configuration when the later candidate arrives: d_M = p_l - p_M (age of the max then, if the max
came first), d_e = p_l - p_e (age of the earlier candidate then), and against (M - S)/M."""
from icommon import *

Z = np.load(HIDE / "r1_I_errors" / "i04b_RT.npz")
eps, pe, pl, pM, M, S = (Z[k] for k in ("eps", "pe", "pl", "pM", "M", "S"))
fin = np.isfinite(eps)
r2 = lambda v, fit: 1 - ((v - fit) ** 2).sum() / ((v - v.mean()) ** 2).sum()


def cell_r2(keys, v):
    uk, inv = np.unique(keys, return_inverse=True)
    cm = np.bincount(inv, v) / np.bincount(inv)
    return r2(v, cm[inv]), len(uk)


print(f"RT sweeps: n = {fin.sum()} with a crossing; sd(eps*) = {eps[fin].std():.4f}")
for name, key in (("(p_e, p_l)", pe * 10 + pl), ("(p_e, p_l, p_M)", pe * 100 + pl * 10 + pM), ("(p_l - p_e, p_l - p_M)", (pl - pe) * 100 + (pl - pM) + 50),
                  ("(p_l - p_e, p_l - p_M, p_l)", (pl - pe) * 1000 + (pl - pM + 50) * 10 + pl)):
    v, k = cell_r2(key[fin], eps[fin])
    print(f"[FIT] cell means by {name}: R^2 = {v:.3f} ({k} cells)")

# within cells (p_e, p_l, p_M): eps* ~ a + b * (M - S)/M
key = (pe * 100 + pl * 10 + pM)
u = (M - S) / M
fit = np.full(len(eps), np.nan); slopes = []
for kk in np.unique(key[fin]):
    s = fin & (key == kk)
    if s.sum() < 30:
        fit[s] = eps[s].mean(); continue
    X = np.c_[np.ones(s.sum()), u[s]]
    beta, *_ = np.linalg.lstsq(X, eps[s], rcond=None)
    fit[s] = X @ beta; slopes.append((kk, beta[0], beta[1], s.sum()))
print(f"[FIT] eps* ~ a + b (M - S)/M within (p_e, p_l, p_M) cells: R^2 = {r2(eps[fin], fit[fin]):.3f}")
sl = np.array(slopes)
print(f"   intercept a: quantiles 10/50/90% = " + " / ".join(f"{v:+.4f}" for v in np.quantile(sl[:, 1], [.1, .5, .9])) +
      ";  slope b: " + " / ".join(f"{v:+.4f}" for v in np.quantile(sl[:, 2], [.1, .5, .9])))
# proportional model: eps* = c * (M - S)/M within cells (no intercept)
fit2 = np.full(len(eps), np.nan)
cs = {}
for kk in np.unique(key[fin]):
    s = fin & (key == kk)
    c = (eps[s] * u[s]).sum() / (u[s] ** 2).sum()
    fit2[s] = c * u[s]; cs[int(kk)] = (c, s.sum())
print(f"[FIT] eps* = c (M - S)/M within cells (one number per cell): R^2 = {r2(eps[fin], fit2[fin]):.3f}")
cv = np.array([v[0] for v in cs.values()]); cn = np.array([v[1] for v in cs.values()])
print(f"   c over cells (weighted by n): quantiles 5/25/50/75/95% = " + " / ".join(f"{v:+.3f}" for v in np.quantile(np.repeat(cv, cn), [.05, .25, .5, .75, .95])))

print("\n## 'M before both': median eps* (and c = eps* M/(M - S)) by d_M = p_l - p_M (rows: age of the max when the later candidate arrives) and d_e = p_l - p_e (columns)")
sel0 = fin & (pM < pe)
rows = []; rows2 = []
for dM in range(2, 10):
    row = [dM]; row2 = [dM]
    for de in range(1, 9):
        s = sel0 & (pl - pM == dM) & (pl - pe == de)
        row.append("." if de >= dM else ("-" if s.sum() < 80 else f"{np.median(eps[s]):+.3f}"))
        row2.append("." if de >= dM else ("-" if s.sum() < 80 else f"{np.median(eps[s] / u[s]):+.3f}"))
    rows.append(row); rows2.append(row2)
print(fmt_table(rows, ["d_M \\ d_e"] + [str(j) for j in range(1, 9)]))
print("\nsame cells, median c = eps* / ((M - S)/M):")
print(fmt_table(rows2, ["d_M \\ d_e"] + [str(j) for j in range(1, 9)]))

print("\n## 'M between' (e, M, l): median eps* by d_M = p_l - p_M (rows) and d_e = p_l - p_e (columns)")
sel0 = fin & (pM > pe) & (pM < pl)
rows = []
for dM in range(1, 9):
    row = [dM]
    for de in range(2, 10):
        s = sel0 & (pl - pM == dM) & (pl - pe == de)
        row.append("." if de <= dM else ("-" if s.sum() < 80 else f"{np.median(eps[s]):+.3f}"))
    rows.append(row)
print(fmt_table(rows, ["d_M \\ d_e"] + [str(j) for j in range(2, 10)]))

print("\n## 'M after both' (e, l, M): median eps* by p_M - p_l (rows) and d_e = p_l - p_e (columns)")
sel0 = fin & (pM > pl)
rows = []
for dM in range(1, 9):
    row = [dM]
    for de in range(1, 9):
        s = sel0 & (pM - pl == dM) & (pl - pe == de)
        row.append("-" if s.sum() < 80 else f"{np.median(eps[s]):+.3f}")
    rows.append(row)
print(fmt_table(rows, ["p_M - p_l \\ d_e"] + [str(j) for j in range(1, 9)]))

# the largest systematic offsets
print("\n## cells (p_e, p_l, p_M) with the largest |median eps*| (n >= 100)")
out = []
for kk in np.unique(key[fin]):
    s = fin & (key == kk)
    if s.sum() >= 100:
        out.append((abs(np.median(eps[s])), int(kk), np.median(eps[s]), np.subtract(*np.quantile(eps[s], [.75, .25])), s.sum()))
out.sort(reverse=True)
rows = [[f"{k // 100}", f"{(k // 10) % 10}", f"{k % 10}", f"{m:+.4f}", f"{iqr:.4f}", n] for _, k, m, iqr, n in out[:14]]
print(fmt_table(rows, ["p_e", "p_l", "p_M", "median eps*", "IQR", "n"]))

# ---------------- MR
Z = np.load(HIDE / "r1_I_errors" / "i04b_MR.npz")
eps, pe, pl, M, x, sag = (Z[k] for k in ("eps", "pe", "pl", "M", "x", "sag"))
fin = np.isfinite(eps)
print(f"\nMR sweeps: n = {fin.sum()}; sd(eps*) = {eps[fin].std():.4f}")
v, k = cell_r2((pe * 10 + pl)[fin], eps[fin]); print(f"[FIT] cell means by (p_e, p_l): R^2 = {v:.3f} ({k} cells)")
# runner-up so far when the later input arrives (largest of the inputs between/before, excluding e), positive part
B = len(x); r = np.arange(B)
xm = x.copy(); xm[r, pe] = -np.inf
for b in range(10):
    xm[pl <= b, b] = -np.inf                 # only inputs before p_l
Sprev = np.maximum(xm.max(1), 0) / M
pSprev = xm.argmax(1)
v, k = cell_r2((pe * 10 + pl)[fin], eps[fin])
key = pe * 10 + pl
fit = np.full(B, np.nan)
for kk in np.unique(key[fin]):
    s = fin & (key == kk)
    X = np.c_[np.ones(s.sum()), Sprev[s]]
    beta, *_ = np.linalg.lstsq(X, eps[s], rcond=None)
    fit[s] = X @ beta
print(f"[FIT] eps* ~ a + b * S_prev+/M within (p_e, p_l) cells: R^2 = {r2(eps[fin], fit[fin]):.3f}")
key3 = pe * 100 + pl * 10 + np.where(Sprev > 0, pSprev, 0)
fit = np.full(B, np.nan)
for kk in np.unique(key3[fin]):
    s = fin & (key3 == kk)
    if s.sum() < 20:
        fit[s] = eps[s].mean(); continue
    X = np.c_[np.ones(s.sum()), Sprev[s]]
    beta, *_ = np.linalg.lstsq(X, eps[s], rcond=None)
    fit[s] = X @ beta
print(f"[FIT] eps* ~ a + b * S_prev+/M within (p_e, p_l, position of the runner-up so far) cells: R^2 = {r2(eps[fin], fit[fin]):.3f}")
