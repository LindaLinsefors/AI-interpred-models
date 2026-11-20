"""S5 follow-up: V[p2] - V[p1] by the actual pair of positions (p1, p2), not only by D = |p1 - p2|.
t = 9, all four comparators on, S > 0.2, p1 < 9, p2 < 9 (the lead's S5 selection).
Shows where the formula 60 (M - 0.87 S)(1 - cos(omega D))/M holds and where it does not.

Usage: python -B x3_pairs.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5200
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 20
GAPS = [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]

cols = []
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    sel = (h[:, COMP] > 0).all(1) & (S[:, 9] > 0.2) & (p1[:, 9] < 9) & (p2[:, 9] < 9)
    s, V, m = scores(h[sel])
    r = np.arange(sel.sum()); P1, P2 = p1[sel, 9], p2[sel, 9]
    cols.append(np.stack([P1, P2, M[sel, 9], S[sel, 9], V[r, P2] - V[r, P1], m[r, P2] - m[r, P1], s[r, P2] - s[r, P1]], 1))
P1, P2, Mt, St, dV, dm, ds = np.concatenate(cols).T
P1 = P1.astype(int); P2 = P2.astype(int)
g = (Mt - St) / Mt; gb = np.digitize(g, GAPS, right=True) - 1
D = np.abs(P1 - P2)
form = 60 * (Mt - 0.87 * St) * (1 - np.cos(OMEGA * D)) / Mt
print(f"seeds {seed0}..{seed0 + nb - 1}; selected {len(P1)}")
np.set_printoptions(precision=1, suppress=True, linewidth=220)


def table(val, k, stat=np.median):
    out = np.full((9, 9), np.nan)
    for a in range(9):
        for b in range(9):
            c = (P1 == a) & (P2 == b) & (gb == k)
            if c.sum() > 100:
                out[a, b] = stat(val[c])
    return out


for k in (3, 1):
    print(f"\n===== gap/M in ({GAPS[k]}, {GAPS[k + 1]}]: rows p1 = 0..8 (position of the max), columns p2 = 0..8 =====")
    print("median (V[p2] - V[p1]) / M:"); print(table(dV / Mt, k))
    print("median of the formula 60 (M - 0.87 S)(1 - cos(omega D))/M:"); print(table(form, k))
    print("ratio:");
    with np.printoptions(precision=2):
        print(table(dV / Mt, k) / table(form, k))
    print("median (m[p2] - m[p1]) / M:"); print(table(dm / Mt, k))
    print("median (s[p2] - s[p1]) / M:"); print(table(ds / Mt, k))
    with np.printoptions(precision=4):
        print("P(s[p2] > s[p1]):"); print(table((ds > 0).astype(float), k, np.mean))
        print("P(m[p2] < m[p1])  (m ranks the max above the runner-up):"); print(table((dm < 0).astype(float), k, np.mean))

print("\n===== restricted to pairs with both positions <= 7 (the range where the readout columns are sinusoidal) =====")
both7 = (P1 <= 7) & (P2 <= 7)
print("median dV/M | formula | ratio, rows D = 1..7, columns gap bins", GAPS)
for Dv in range(1, 8):
    row = []
    for k in range(len(GAPS) - 1):
        c = both7 & (D == Dv) & (gb == k)
        row.append(f"{np.median((dV / Mt)[c]):6.1f}|{np.median(form[c]):5.1f}|{np.median((dV / Mt)[c]) / np.median(form[c]):5.2f}" if c.sum() > 50 else "      -      ")
    print(Dv, "   ".join(row))
print("pairs involving position 8: median dV/M | formula | ratio")
for Dv in range(1, 9):
    row = []
    for k in range(len(GAPS) - 1):
        c = ~both7 & (D == Dv) & (gb == k)
        row.append(f"{np.median((dV / Mt)[c]):6.1f}|{np.median(form[c]):5.1f}|{np.median((dV / Mt)[c]) / np.median(form[c]):5.2f}" if c.sum() > 50 else "      -      ")
    print(Dv, "   ".join(row))

# how well does the formula predict dV/M sequence by sequence?
print("\nsequence-level: R^2 (1 - SS_res/SS_tot, centred) of the formula as a predictor of dV/M, and the best single scale factor")
for nm, c in (("all pairs", np.ones(len(P1), bool)), ("both <= 7", both7), ("both <= 7, D = 2..5", both7 & (D >= 2) & (D <= 5)),
              ("both <= 7, D = 1", both7 & (D == 1)), ("both <= 7, D = 6, 7", both7 & (D >= 6))):
    y = (dV / Mt)[c]; f = form[c]
    b = (y * f).sum() / (f * f).sum()
    print(f"  {nm:22s} n = {c.sum():7d}  R^2 = {1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}   best scale = {b:.3f}  "
          f"R^2 with it = {1 - ((y - b * f) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}   corr = {np.corrcoef(y, f)[0, 1]:.3f}")
