"""Attacks on the S2/S3/S5 interpretation ("V is a cosine wave centred on the max, amplitude ~ M - 0.87 S").

All at t = 9, sequences with all four comparators on and S_9 > 0.1 (the lead's S2 selection).
 A. Shuffled ages: permute the a1 labels and redo the lead's regression + alignment on d.
 B. Model comparison per a1 (pooled centred R^2 on i <= 7): regressors (M-S, S, x9) [lead], (M, x9), (M-S, x9),
    and the lead's design fitted separately in every (a1, a2) cell.
 C. Residuals by position.
 D. Complex amplitude of V[0..7] at omega per sequence, Z (V[i] ~ Re(Z e^{i omega i}) + c). K := -Z e^{i omega p1}
    is real and positive for a trough at the max. Per (a1, a2) cell: K ~ alpha (M-S) + beta S + gamma x9.
    Is beta a fixed fraction of alpha (lead: beta/alpha = 0.13), or does it turn with the runner-up's position?
 E. Amplitude law without a regression: mean of Re K / M in bins of S/M.
 F. The literal model V[i] = -K0 (M - c S) cos(omega (i - p1)) on i <= 7.

Usage: python -B x2_regression_attacks.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *
from s2_s3 import by_d, fit, cosfit

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5100
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 20

Vs, cols = [], []
n_tot = 0
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    sel = (h[:, COMP] > 0).all(1) & (S[:, 9] > 0.1)
    n_tot += len(x)
    Vs.append(scores(h[sel])[1])
    cols.append(np.stack([M[sel, 9], S[sel, 9], x[sel, 9], p1[sel, 9], p2[sel, 9]], 1))
V = np.concatenate(Vs); Mt, St, x9, P1, P2 = np.concatenate(cols).T
P1 = P1.astype(int); P2 = P2.astype(int); a1 = 9 - P1; a2 = 9 - P2
X = np.stack([Mt - St, St, x9], 1)
print(f"seeds {seed0}..{seed0 + nb - 1}; {n_tot} sequences; selected {len(V)} (all four comparators on at t = 9, S > 0.1)")
keep = a1 >= 2
print(f"a1 >= 2: {keep.sum()} rows")

# ---------------------------------------------------------------- A. shuffled ages
print("\n=== A. shuffled ages (a1 labels permuted among the selected sequences with a1 >= 2) ===")
rng = np.random.default_rng(seed0 + 777)
idx = np.flatnonzero(keep)
a1_sh = a1.copy(); a1_sh[idx] = a1[idx][rng.permutation(len(idx))]
A_true, A_sh, B_true, B_sh = {}, {}, {}, {}
for a in range(2, 10):
    A_true[a], B_true[a], _ = fit(X[a1 == a], V[a1 == a])[0]
    A_sh[a], B_sh[a], _ = fit(X[a1_sh == a], V[a1_sh == a])[0]
dt, dsh, dbt, dbs = by_d(A_true), by_d(A_sh), by_d(B_true), by_d(B_sh)
print("d:            " + " ".join(f"{d:+6d}" for d in range(-7, 8)))
print("A true ages:  " + " ".join(f"{dt[d][0]:6.1f}" for d in range(-7, 8)))
print("A shuffled:   " + " ".join(f"{dsh[d][0]:6.1f}" for d in range(-7, 8)))
print("B true ages:  " + " ".join(f"{dbt[d][0]:6.1f}" for d in range(-7, 8)))
print("B shuffled:   " + " ".join(f"{dbs[d][0]:6.1f}" for d in range(-7, 8)))
print("A_i with shuffled ages, by index i = 0..9 (same for every label, up to noise):")
for a in (2, 5, 9):
    print(f"  label {a}: " + " ".join(f"{v:6.1f}" for v in A_sh[a]))
print("  (A_i of ONE regression pooled over all a1 >= 2):", " ".join(f"{v:6.1f}" for v in fit(X[keep], V[keep])[0][0]))
for nm, tab in (("true", A_true), ("shuffled", A_sh)):
    amps = [cosfit(tab[a][:8], np.arange(8.0))[0] for a in tab]
    print(f"  cosine amplitude of A_i over i = 0..7 per label, {nm}: " + " ".join(f"{v:5.1f}" for v in amps))

# ---------------------------------------------------------------- B. model comparison, C. residuals
print("\n=== B. model comparison: pooled centred R^2 over i = 0..7 (and over i = 0..9), per a1 ===")
print("a1      n     lead (M-S,S,x9)    (M, x9)      (M-S, x9)    lead design per (a1,a2) cell     [i = 0..9 in brackets]")
res_cell = np.full(V.shape, np.nan); res_lead = np.full(V.shape, np.nan)
for a in range(2, 10):
    k = a1 == a
    out = []
    Y = V[k]; ssc = ((Y - Y.mean(0)) ** 2).sum(0)
    for Xk in (X[k], np.stack([Mt[k], x9[k]], 1), np.stack([(Mt - St)[k], x9[k]], 1)):
        ssr = fit(Xk, Y)[1]
        out.append((1 - ssr[:8].sum() / ssc[:8].sum(), 1 - ssr.sum() / ssc.sum()))
    res_lead[k] = Y - X[k] @ fit(X[k], Y)[0]
    for b in range(0, 10):
        kk = k & (a2 == b)
        if kk.sum() > 100:
            res_cell[kk] = V[kk] - X[kk] @ fit(X[kk], V[kk])[0]
    got = ~np.isnan(res_cell[k][:, 0])                   # rows in cells large enough to be fitted (all but a few)
    ssr = (res_cell[k][got] ** 2).sum(0); sscg = ((Y[got] - Y[got].mean(0)) ** 2).sum(0)
    out.append((1 - ssr[:8].sum() / sscg[:8].sum(), 1 - ssr.sum() / sscg.sum()))
    print(f"{a}  {k.sum():7d}   " + "    ".join(f"{u:.3f} [{v:.3f}]" for u, v in out))

print("\n=== C. residuals of the lead's regression by position (rms in score units; rms of V centred; ratio) ===")
print("index i:                 " + " ".join(f"{i:6d}" for i in range(10)))
for a in range(2, 10):
    k = a1 == a
    r = np.sqrt((res_lead[k] ** 2).mean(0)); sd = V[k].std(0)
    print(f"a1={a} rms residual:       " + " ".join(f"{v:6.2f}" for v in r))
    print(f"      sd of V[i]:         " + " ".join(f"{v:6.2f}" for v in sd))
    print(f"      rms resid per-cell: " + " ".join(f"{v:6.2f}" for v in np.sqrt((res_cell[k] ** 2).mean(0))))
k = keep
print("all a1 >= 2: residual rms / sd by index (lead design):   " + " ".join(f"{v:.3f}" for v in np.sqrt((res_lead[k] ** 2).mean(0)) / V[k].std(0)))
print("median M in the selection:", np.median(Mt[k]), " -> residual rms in units of median M:",
      " ".join(f"{v:.2f}" for v in np.sqrt((res_lead[k] ** 2).mean(0)) / np.median(Mt[k])))

# ---------------------------------------------------------------- D. complex amplitude per cell
print("\n=== D. complex amplitude K = -Z e^{i omega p1} of V[0..7]; per (a1, a2): K ~ alpha (M-S) + beta S + gamma x9 ===")
pos = np.arange(8.0)
G = np.stack([np.cos(OMEGA * pos), np.sin(OMEGA * pos), np.ones(8)], 1)
co = V[:, :8] @ np.linalg.pinv(G).T                       # (n, 3): a, b, c
Z = co[:, 0] - 1j * co[:, 1]
fitted = co @ G.T
frac = 1 - ((V[:, :8] - fitted) ** 2).sum() / ((V[:, :8] - V[:, :8].mean(1, keepdims=True)) ** 2).sum()
print(f"fraction of the per-state variance of V over i = 0..7 that is an omega-sinusoid (pooled): {frac:.4f}")
K = -Z * np.exp(1j * OMEGA * P1)
alpha = np.full((10, 10), np.nan + 0j); beta = alpha.copy(); gamma = alpha.copy(); r2cell = np.full((10, 10), np.nan)
ncell = np.zeros((10, 10), int)
for a in range(2, 10):
    for b in range(0, 10):
        kk = (a1 == a) & (a2 == b)
        ncell[a, b] = kk.sum()
        if kk.sum() < 300:
            continue
        cf = np.linalg.lstsq(X[kk], np.stack([K[kk].real, K[kk].imag], 1), rcond=None)[0]
        alpha[a, b], beta[a, b], gamma[a, b] = cf[:, 0] + 1j * cf[:, 1]
        r = K[kk] - X[kk] @ (cf[:, 0] + 1j * cf[:, 1])
        r2cell[a, b] = 1 - (np.abs(r) ** 2).sum() / (np.abs(K[kk]) ** 2).sum()
np.set_printoptions(precision=2, suppress=True, linewidth=220)
print("|alpha| (rows a1 = 2..9, columns a2 = 0..9):"); print(np.abs(alpha[2:]))
print("arg alpha in units of omega (= offset of the trough from p1, in positions; + = trough EARLIER than p1):"); print(np.angle(alpha[2:]) / OMEGA)
print("Re(beta/alpha):"); print((beta / alpha)[2:].real)
print("Im(beta/alpha):"); print((beta / alpha)[2:].imag)
print("|beta|:"); print(np.abs(beta[2:]))
print("uncentred complex R^2 per cell:"); print(r2cell[2:])
print("n per cell:"); print(ncell[2:])
ok = ~np.isnan(alpha.real)
ok[:, 0] = False                                           # a2 = 0: S = x9, beta and gamma are not separately identified
aa, bb = np.nonzero(ok)
rot = np.exp(1j * OMEGA * ((9 - aa) - (9 - bb)))           # e^{i omega (p1 - p2)}: K of a unit trough at p2
Gc = np.stack([np.ones(len(aa)), rot], 1)
wts = np.sqrt(ncell[aa, bb])
b01 = np.linalg.lstsq(Gc * wts[:, None], beta[aa, bb] * wts, rcond=None)[0]
pred = Gc @ b01
print(f"\nfit over cells (a2 >= 1), weighted by sqrt(n): beta ~ b0 + b1 e^(i omega (p1 - p2)):  b0 = {b01[0]:.2f}, b1 = {b01[1]:.2f}"
      f"   (b0: trough at the max per unit S; b1 > 0: trough at the runner-up, b1 < 0: crest at the runner-up)")
print(f"  weighted share of sum |beta - mean|^2 explained by the b1 term: "
      f"{1 - (wts ** 2 * np.abs(beta[aa, bb] - pred) ** 2).sum() / (wts ** 2 * np.abs(beta[aa, bb] - np.average(beta[aa, bb], weights=wts ** 2)) ** 2).sum():.3f}")
print(f"  weighted mean alpha = {np.average(alpha[aa, bb], weights=wts ** 2):.2f}; weighted mean beta = {np.average(beta[aa, bb], weights=wts ** 2):.2f}"
      f"; mean beta / mean alpha = {np.average(beta[aa, bb], weights=wts ** 2) / np.average(alpha[aa, bb], weights=wts ** 2):.3f}")
print("  beta by signed separation p2 - p1 (weighted mean over cells): ")
for sd in range(-7, 8):
    c = ((9 - bb) - (9 - aa)) == sd
    if c.any():
        print(f"    p2 - p1 = {sd:+d}: beta = {np.average(beta[aa, bb][c], weights=wts[c] ** 2):.2f}   alpha = {np.average(alpha[aa, bb][c], weights=wts[c] ** 2):.2f}"
              f"   Re(beta/alpha) = {np.average((beta / alpha)[aa, bb][c].real, weights=wts[c] ** 2):+.3f}")

# ---------------------------------------------------------------- E. amplitude law without regression
print("\n=== E. mean of Re K / M and Im K / M in bins of S/M (a1 >= 2, a2 >= 1), by a1; then a line fit Re K/M ~ k0 - k1 S/M ===")
edges = np.linspace(0, 1, 11)
rr = St / Mt
kk0 = keep & (a2 >= 1)
print("S/M bin:      " + " ".join(f"{(edges[i] + edges[i + 1]) / 2:6.2f}" for i in range(10)))
for a in list(range(2, 10)) + ["all"]:
    k = kk0 if a == "all" else kk0 & (a1 == a)
    bins = np.digitize(rr[k], edges) - 1
    re = [np.mean((K.real / Mt)[k][bins == i]) if (bins == i).sum() > 200 else np.nan for i in range(10)]
    im = [np.mean((K.imag / Mt)[k][bins == i]) if (bins == i).sum() > 200 else np.nan for i in range(10)]
    Gl = np.stack([np.ones(k.sum()), rr[k], (x9 / Mt)[k]], 1)
    c0 = np.linalg.lstsq(Gl, (K.real / Mt)[k], rcond=None)[0]
    print(f"a1={a}: Re " + " ".join(f"{v:6.1f}" for v in re) + f"   | fit: K/M = {c0[0]:.1f} (1 - {-c0[1] / c0[0]:.3f} S/M) + {c0[2]:.1f} x9/M")
    print(f"       Im " + " ".join(f"{v:6.1f}" for v in im))

# ---------------------------------------------------------------- F. literal model
print("\n=== F. literal model on i = 0..7: V[i] - mean_i ~ -K0 (M - c S) (cos(omega (i - p1)) - mean_i) ===")
k = keep
Vc = V[k][:, :8] - V[k][:, :8].mean(1, keepdims=True)
cw = np.cos(OMEGA * (pos[None, :] - P1[k][:, None])); cw = cw - cw.mean(1, keepdims=True)
sst = (Vc ** 2).sum()
for K0, c in ((60, 0.87), (60, 1.0), (60, 0.0)):
    pred = -K0 * (Mt[k] - c * St[k])[:, None] * cw
    print(f"K0 = {K0}, c = {c}: pooled R^2 (per-state centred over i <= 7) = {1 - ((Vc - pred) ** 2).sum() / sst:.3f}")
# best K0, c by least squares: Vc ~ -(kM M - kS S) cw
D1 = -(Mt[k][:, None] * cw).ravel(); D2 = (St[k][:, None] * cw).ravel()
cf = np.linalg.lstsq(np.stack([D1, D2], 1), Vc.ravel(), rcond=None)[0]
pred = cf[0] * D1 + cf[1] * D2
print(f"best fit: K0 = {cf[0]:.1f}, c = {cf[1] / cf[0]:.3f}: R^2 = {1 - ((Vc.ravel() - pred) ** 2).sum() / sst:.3f}")
for a in range(2, 10):
    ka = a1[k] == a
    cf = np.linalg.lstsq(np.stack([D1.reshape(-1, 8)[ka].ravel(), D2.reshape(-1, 8)[ka].ravel()], 1), Vc[ka].ravel(), rcond=None)[0]
    pred = (cf[0] * D1.reshape(-1, 8)[ka] + cf[1] * D2.reshape(-1, 8)[ka])
    lit = -60 * (Mt[k][ka] - 0.87 * St[k][ka])[:, None] * cw[ka]
    print(f"  a1={a}: best K0 = {cf[0]:5.1f}, c = {cf[1] / cf[0]:.3f}, R^2 = {1 - ((Vc[ka] - pred) ** 2).sum() / (Vc[ka] ** 2).sum():.3f};"
          f"  literal (60, 0.87): R^2 = {1 - ((Vc[ka] - lit) ** 2).sum() / (Vc[ka] ** 2).sum():.3f}")
