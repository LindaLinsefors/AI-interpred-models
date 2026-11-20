"""Reviewer N3: small checks of numbers in sections 6-10 of report.md. Read-only on everything else.
Run:  python -B n3_checks.py > n3_checks_out.txt
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from common import w, W, U, run, running_stats, COMP, ORD, sample
from modes import RIGHT

np.set_printoptions(precision=3, suppress=True, linewidth=220)
W4 = w[4]


def final_h(x):
    h = np.zeros((x.shape[0], 16))
    for t in range(10):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
    return h


def top3(x):
    o = np.argsort(-x, axis=1)
    pM, pS, pT = o[:, 0], o[:, 1], o[:, 2]
    r = np.arange(len(x))
    return pM, pS, pT, x[r, pM], x[r, pS], x[r, pT]


# ---------------------------------------------------------------- A: margin runner-up over max
print("=== A. median of (s[pS] - s[pM])/M at t = 9 (S > 0.2, both older than x_9, all four comparators on)")
bins = [0.0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]
rows = []
for seed in range(7000, 7010):
    x = sample(400_000, seed)
    h = final_h(x)
    L = h @ U.T
    pM, pS, pT, M, S, T = top3(x)
    r = np.arange(len(x))
    ok = (S > 0.2) & (pM < 9) & (pS < 9) & (h[:, COMP] > 0).all(1)
    rows.append(np.c_[np.abs(pM - pS), (M - S) / M, (L[r, pS] - L[r, pM]) / M, 9 - pM, (pS > pM)][ok])
A = np.concatenate(rows)
print("rows: D = 1..8; columns: gap/M in", bins)
tab = np.full((8, 6), np.nan)
for D in range(1, 9):
    for b in range(6):
        m = (A[:, 0] == D) & (A[:, 1] > bins[b]) & (A[:, 1] <= bins[b + 1])
        if m.sum() > 300:
            tab[D - 1, b] = np.median(A[m, 2])
print(np.round(tab, 1))
print("range of the medians for gap/M > 0.05: %.1f .. %.1f" % (np.nanmin(tab[:, 1:]), np.nanmax(tab[:, 1:])))
print("by age of the max (rows 1..9) and the same gap bins (pooled over D):")
tab = np.full((9, 6), np.nan)
for a in range(1, 10):
    for b in range(6):
        m = (A[:, 3] == a) & (A[:, 1] > bins[b]) & (A[:, 1] <= bins[b + 1])
        if m.sum() > 300:
            tab[a - 1, b] = np.median(A[m, 2])
print(np.round(tab, 1))
print("ages >= 2, gap/M > 0.05: range of medians per (age, bin): %.1f .. %.1f" % (np.nanmin(tab[1:, 1:]), np.nanmax(tab[1:, 1:])))
del A, rows

# ---------------------------------------------------------------- B: worst arrangements (max last)
print("\n=== B. P(answer = pT) by arrangement (pS, pT, pM), max after both, T > 0.2; gap (S-T)/S bins (0,.03],(.03,.08],(.08,.2],(.2,.5]")
gb = [0.0, 0.03, 0.08, 0.2, 0.5]
cnt = np.zeros((10, 10, 10, 4))
hit = np.zeros((10, 10, 10, 4))
for seed in range(7100, 7130):
    x = sample(400_000, seed)
    ans = (final_h(x) @ U.T).argmax(1)
    pM, pS, pT, M, S, T = top3(x)
    ok = (pM > pS) & (pM > pT) & (T > 0.2)
    g = (S - T) / S
    b = np.digitize(g, gb, right=True) - 1
    ok &= (b >= 0) & (b < 4)
    np.add.at(cnt, (pS[ok], pT[ok], pM[ok], b[ok]), 1)
    np.add.at(hit, (pS[ok], pT[ok], pM[ok], b[ok]), (ans == pT)[ok])
rate = hit / np.maximum(cnt, 1)
for b in range(4):
    flat = [(rate[i, j, k, b], cnt[i, j, k, b], (i, j, k)) for i in range(10) for j in range(10) for k in range(10) if cnt[i, j, k, b] >= 150]
    flat.sort(reverse=True)
    print(f"bin {b+1}: top 10 arrangements (rate, n, (pS,pT,pM)):", [(round(r_, 3), int(n_), c_) for r_, n_, c_ in flat[:10]])
for c_ in [(0, 7, 9), (8, 0, 9), (7, 0, 9), (6, 5, 9), (7, 6, 9), (7, 8, 9), (8, 1, 9), (7, 0, 8), (0, 8, 9)]:
    print(c_, np.round(rate[c_], 3), cnt[c_].astype(int))

# ---------------------------------------------------------------- C: runner vector and centre per unit of S
print("\n=== C. runner right eigenvector (modes.py gauge) on ORD units, and w4*5.2*R_runner (centre shift per unit S if runner = w4*(r0 M + 5.2 S))")
Rr = np.real(RIGHT["runner"])
for u, v in zip(ORD, Rr):
    print(f"unit {u:2d}: R_runner = {v:+.3f}   w4*5.2*R = {W4*5.2*v:+.2f}")
x = sample(400_000, 7200)
H = run(x)
st = running_stats(x)
M1, M2, a1 = st["M1"], st["M2"], st["a1"]
Sp = np.maximum(M2, 0)
print("measured centre (mean over ages 2..9 of the mean of h_u/M) in bins of S+/M <=0.1 | 0.45-0.55 | 0.85-0.95, and slope per unit S between first and last bin")
cent = np.zeros((3, 16))
for bi, (lo, hi) in enumerate([(-1, 0.1), (0.45, 0.55), (0.85, 0.95)]):
    per_age = []
    for a in range(2, 10):
        m = (a1 == a) & (M1 > 0.3) & (Sp / M1 > lo) & (Sp / M1 <= hi)
        per_age.append((H[m] / M1[m][:, None]).mean(0))
    cent[bi] = np.mean(per_age, 0)
smid = [0.03, 0.5, 0.9]
for u in ORD:
    print(f"unit {u:2d}: centre {cent[0,u]:6.2f} {cent[1,u]:6.2f} {cent[2,u]:6.2f}   slope per unit S: {(cent[2,u]-cent[0,u])/(smid[2]-smid[0]):+.2f}")
del H

# ---------------------------------------------------------------- D: unit 4 false firings for new runners-up, by age of the max when the input arrives
print("\n=== D. new runner-up at step t (S_{t-1}+ < x_t < M_{t-1}, M_{t-1} > 0): P(unit 4 fires) and median theta_4/M, by age of the max at arrival t - p1 = 1..9")
fire = np.zeros(10); tot = np.zeros(10)
th_list = [[] for _ in range(10)]
cut_none = np.zeros(10); allzero = np.zeros((2, 10)); nrec = np.zeros((2, 10))
kap_rows = []
for seed in range(7300, 7305):
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    for t in range(1, 10):
        Mp, Sprev, pprev = st["M1"][:, t - 1], np.maximum(st["M2"][:, t - 1], 0), st["p1"][:, t - 1]
        age = t - pprev
        th4 = x[:, t] - Z[:, t, 4] / W4                 # theta_4(t)
        nr = (x[:, t] > Sprev) & (x[:, t] < Mp) & (Mp > 0)
        np.add.at(tot, age[nr], 1); np.add.at(fire, age[nr], (H[nr, t, 4] > 0))
        np.add.at(cut_none, age[nr], (H[nr, t][:, COMP] > 0).all(1))
        sel = (Mp > 0.3)
        for a in range(1, 10):
            mm = sel & (age == a)
            if mm.any():
                th_list[a].append((th4[mm] / Mp[mm])[:20000])
        kap_rows.append(np.c_[age, Mp, Sprev, Mp - th4][sel])
        # new maximum over a positive old maximum: all four comparators zero?
        rec = (x[:, t] > Mp) & (Mp > 0)
        z4 = (H[:, t][:, COMP] == 0).all(1)
        nrec[0, t] += rec.sum(); allzero[0, t] += (rec & z4).sum()
        rec2 = (x[:, t] > Mp) & (Mp > 0.3) & (pprev < t - 1)
        nrec[1, t] += rec2.sum(); allzero[1, t] += (rec2 & z4).sum()
print("P(unit 4 fires | new runner-up), ages 1..9:", np.round(fire[1:] / tot[1:], 3))
print("P(no comparator cut | new runner-up), ages 1..9:", np.round(cut_none[1:] / tot[1:], 3))
print("median theta_4(t)/M_{t-1}, ages 1..9:", np.round([np.median(np.concatenate(th_list[a])) for a in range(1, 10)], 3))
K = np.concatenate(kap_rows)
kk = []
for a in range(1, 10):
    m = K[:, 0] == a
    X = np.c_[K[m, 1], K[m, 1] - K[m, 2]]
    kk.append(np.linalg.lstsq(X, K[m, 3], rcond=None)[0])
print("M - theta_4 = eps*M + kappa*(M - S+): kappa by age at arrival 1..9:", np.round([k_[1] for k_ in kk], 3), " eps:", np.round([k_[0] for k_ in kk], 3))
print("all four comparators exactly zero at a new maximum, by step t = 1..9")
print("  (i) old max > 0:                         ", np.round(allzero[0, 1:] / nrec[0, 1:], 3), " pooled t=1..8: %.3f" % (allzero[0, 1:9].sum() / nrec[0, 1:9].sum()))
print("  (ii) old max > 0.3 and not at step t-1:   ", np.round(allzero[1, 2:] / nrec[1, 2:], 3), "(t = 2..9)")
del H, Z

# ---------------------------------------------------------------- E: shift experiment on uniform inputs
print("\n=== E. accuracy on Unif(-1,1) + c  (2^20 sequences each)")
for c in [-1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2]:
    x = sample(2 ** 20, 7400, "uniform") + c
    acc = ((final_h(x) @ U.T).argmax(1) == np.argsort(x, 1)[:, -2]).mean()
    print(f"c = {c:+.2f}: accuracy {acc:.4f}")
