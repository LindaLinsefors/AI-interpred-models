"""Attacks on S1 ("m = readout without the comparators ranks the MAXIMUM first").

 a. t = 9: which index does m rank first, by position of the max (index 0 is only ever read at t = 9).
 b. The current input (index 9): how often is it the largest entry of m, and how large is m[9] - m[p1]?
 c. Specificity: remove four OTHER units (every 4-subset of the 9 wave units) instead of the comparators.
 d. Regression-free test of "V is centred on the max": P(argmin V = p1), P(argmax V = p2), by age of the max.
 e. Margin of the max in m, in units of M, by age and gap.

Usage: python -B x1_m_rank.py [seed0] [n_batches]
"""
import sys
from itertools import combinations

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5300
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ts = range(2, 10)
subsets = list(combinations(WAVE, 4))

win9 = np.zeros((10, 9))                    # t = 9: [p1, argmax index of m over 0..8]
n9 = np.zeros(10); p2win9 = np.zeros(10)
pair9 = np.zeros((9, 9)); npair9 = np.zeros((9, 9))      # t = 9: P(m[p1] > m[i]) by (p1, i)
n = {t: 0 for t in ts}; cur_first = dict(n); cur_margin = {t: [] for t in ts}; cur_x = {t: [] for t in ts}
hit_sub = np.zeros((len(subsets), 10)); hit_sub2 = np.zeros((len(subsets), 10))
hit_comp = np.zeros(10); hit_comp2 = np.zeros(10); hit_full = np.zeros(10); hit_full2 = np.zeros(10); n_t = np.zeros(10)
vmin = np.zeros((10, 10)); vmax2 = np.zeros((10, 10)); vn = np.zeros((10, 10))      # [t, a1], all four on, S > 0.1, a1 >= 1
vmin7 = np.zeros(10); vn7 = np.zeros(10); vmin7_pm1 = np.zeros(10)                 # t = 9, indices 0..7 only, by a1
marg = {}                                                                          # (a1, gap bin) -> list of margins / M

GAPS = [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]
for x, H, M, S, p1, p2 in batches(nb, seed0):
    rows = np.arange(len(x))
    for t in ts:
        h = H[:, t]
        s, V, m = scores(h)
        P1, P2, Mt, St = p1[:, t], p2[:, t], M[:, t], S[:, t]
        c = (St > 0) & (P1 < t)
        old = slice(9 - t, 9)
        am = m[:, old].argmax(1)
        n[t] += c.sum()
        cur_first[t] += (m[:, 9 - t:].argmax(1) == t)[c].sum()
        cur_margin[t].append(((m[:, 9] - m[rows, P1 + 9 - t]) / Mt)[c][:20000]); cur_x[t].append((x[:, t] / Mt)[c][:20000])
        # c. specificity
        n_t[t] += c.sum()
        hit_comp[t] += (am == P1)[c].sum(); hit_comp2[t] += (am == P2)[c].sum()
        asf = s[:, old].argmax(1); hit_full[t] += (asf == P1)[c].sum(); hit_full2[t] += (asf == P2)[c].sum()
        if t in (5, 9):
            for k, sub in enumerate(subsets):
                sub = list(sub)
                alt = (s - h[:, sub] @ U[:, sub].T)[:, old].argmax(1)
                hit_sub[k, t] += (alt == P1)[c].sum(); hit_sub2[k, t] += (alt == P2)[c].sum()
        # d. V centred on the max?
        on = (h[:, COMP] > 0).all(1) & (St > 0.1) & (P1 < t)
        a1 = t - P1
        np.add.at(vn[t], a1[on], 1)
        np.add.at(vmin[t], a1[on], (V[:, old].argmin(1) == P1)[on])
        np.add.at(vmax2[t], a1[on], (V[:, old].argmax(1) == P2)[on])
        if t == 9:
            o7 = on & (P1 <= 7)
            am7 = V[:, :8].argmin(1)
            np.add.at(vn7, a1[o7], 1); np.add.at(vmin7, a1[o7], (am7 == P1)[o7]); np.add.at(vmin7_pm1, a1[o7], (np.abs(am7 - P1) <= 1)[o7])
            # a.
            np.add.at(win9, (P1[c], am[c]), 1); np.add.at(n9, P1[c], 1); np.add.at(p2win9, P1[c], (am == P2)[c])
            for i in range(9):
                cc = c & (P1 != i)
                np.add.at(npair9[:, i], P1[cc], 1); np.add.at(pair9[:, i], P1[cc], (m[rows, P1] > m[:, i])[cc])
        # e. margin of the max in m among older inputs
        P1c = np.minimum(P1, t - 1)                    # rows with p1 = t are masked out by c below
        mm = m[:, old].copy(); top = mm[rows, P1c]; mm[rows, P1c] = -np.inf
        margin = (top - mm.max(1)) / Mt
        gbin = np.digitize((Mt - St) / Mt, GAPS, right=True) - 1
        for a in range(1, t + 1):
            for g in (0, 1, 3):
                k = c & (a1 == a) & (gbin == g)
                marg.setdefault((a, g), []).append(margin[k][:5000])

np.set_printoptions(precision=3, suppress=True, linewidth=220)
print(f"seeds {seed0}..{seed0 + nb - 1}; {nb * B_DEFAULT} sequences")
print("\n=== a. t = 9, S > 0, p1 < 9: distribution of argmax m over indices 0..8 (columns) by position of the max (rows p1 = 0..8) ===")
print(win9[:9] / n9[:9, None])
print("P(argmax m = p1) by p1:", np.diag(win9[:9]) / n9[:9])
print("P(argmax m = p2) by p1:", p2win9[:9] / n9[:9])
print("P(m[p1] > m[i]) by p1 (rows) and other index i (columns 0..8):")
with np.errstate(invalid="ignore"):
    print(pair9 / npair9)

print("\n=== b. the current input in m (S_t > 0, p1 < t): P(index 9 is the largest entry of m over indices 9-t..9); median (m[9] - m[p1])/M ===")
for t in ts:
    d = np.concatenate(cur_margin[t]); xr = np.concatenate(cur_x[t])
    b = np.polyfit(xr, d, 1)
    print(f"t={t}: P(current input first in m) = {cur_first[t] / n[t]:.4f};  median (m[9]-m[p1])/M = {np.median(d):7.2f};  "
          f"linear fit (m[9]-m[p1])/M ~ {b[1]:.1f} + {b[0]:.1f} x_t/M   corr = {np.corrcoef(xr, d)[0, 1]:.3f};  "
          f"P(m[9] > m[p1]) for x_t/M < 0: {np.mean(d[xr < 0] > 0):.3f}, for 0 < x_t/M < 0.5: {np.mean(d[(xr > 0) & (xr < 0.5)] > 0):.3f}, "
          f"for x_t/M > 0.5: {np.mean(d[xr > 0.5] > 0):.3f}")

print("\n=== c. specificity (S_t > 0, p1 < t; argmax over older inputs) ===")
print("t   full score s: P(=p1) P(=p2)  |  without comparators: P(=p1) P(=p2)")
for t in ts:
    print(f"{t}   {hit_full[t] / n_t[t]:.4f} {hit_full2[t] / n_t[t]:.4f}   |   {hit_comp[t] / n_t[t]:.4f} {hit_comp2[t] / n_t[t]:.4f}")
for t in (5, 9):
    q = hit_sub[:, t] / n_t[t]; q2 = hit_sub2[:, t] / n_t[t]
    best = q.argmax()
    print(f"t={t}: removing four WAVE units instead (all {len(subsets)} subsets): P(argmax = p1): min {q.min():.4f}, median {np.median(q):.4f}, "
          f"max {q.max():.4f} (subset {subsets[best]}, P(=p2) there {q2[best]:.4f});  P(argmax = p2): min {q2.min():.4f}, median {np.median(q2):.4f}, max {q2.max():.4f}")
    k = subsets.index((10, 11, 12, 15))
    print(f"      subset (10, 11, 12, 15): P(=p1) = {q[k]:.4f}, P(=p2) = {q2[k]:.4f}")

print("\n=== d. V itself (all four comparators on, S_t > 0.1, p1 < t): P(argmin V over older inputs = p1) by a1 = 1..9, rows t = 2..9 ===")
for t in ts:
    print(t, " ".join(f"{vmin[t, a] / vn[t, a]:.3f}" if vn[t, a] else "  -  " for a in range(1, 10)))
print("pooled over t:", " ".join(f"{vmin[:, a].sum() / vn[:, a].sum():.3f}" for a in range(1, 10)))
print("P(argmax V over older inputs = p2) by a1, rows t:")
for t in ts:
    print(t, " ".join(f"{vmax2[t, a] / vn[t, a]:.3f}" if vn[t, a] else "  -  " for a in range(1, 10)))
print("t = 9, indices 0..7 only: P(argmin V = p1) by a1 = 2..9:", " ".join(f"{vmin7[a] / vn7[a]:.3f}" for a in range(2, 10)))
print("t = 9, indices 0..7 only: P(|argmin V - p1| <= 1) by a1 = 2..9:", " ".join(f"{vmin7_pm1[a] / vn7[a]:.3f}" for a in range(2, 10)))

print("\n=== e. margin of the max in m over the best other older input, in units of M: median and P(margin > 0) by a1; gap bins (0,.05], (.05,.15], (.3,.5] ===")
for a in range(1, 10):
    out = []
    for g in (0, 1, 3):
        d = np.concatenate(marg[(a, g)])
        out.append(f"median {np.median(d):6.2f}  P>0 {np.mean(d > 0):.3f}")
    print(f"a1={a}:  " + "   |   ".join(out))
