"""Selection check for S2/S5: those claims condition on 'all four comparators on at t = 9'.
What happens in the complement?  t = 9, S > 0.2, p1 < 9, p2 < 9, split by the number of comparators on.

Usage: python -B x8_selection.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5800
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rows = []
tot = 0; n_on_all = np.zeros(5); n_new_ru = np.zeros(5); n_new_max = np.zeros(5)
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    non = (h[:, COMP] > 0).sum(1)
    tot += len(x)
    n_on_all += np.bincount(non, minlength=5)
    n_new_ru += np.bincount(non[p2[:, 9] == 9], minlength=5); n_new_max += np.bincount(non[p1[:, 9] == 9], minlength=5)
    sel = (S[:, 9] > 0.2) & (p1[:, 9] < 9) & (p2[:, 9] < 9)
    s, V, m = scores(h[sel])
    r = np.arange(sel.sum()); P1, P2, Mt = p1[sel, 9], p2[sel, 9], M[sel, 9]
    rows.append(np.stack([non[sel], (V[r, P2] - V[r, P1]) / Mt, (m[r, P2] - m[r, P1]) / Mt, (s[r, P2] - s[r, P1]) / Mt,
                          s.argmax(1) == P2, x[sel, 9] / S[sel, 9], (Mt - S[sel, 9]) / Mt], 1))
d = np.concatenate(rows)
non, dV, dm, ds, corr, xr, gap = d.T
print(f"seeds {seed0}..{seed0 + nb - 1}; {tot} sequences")
print("number of comparators on at t = 9 (0..4), share of ALL sequences:", np.round(n_on_all / tot, 4))
print("  when x_9 is the new max (p1 = 9):      ", np.round(n_new_max / n_new_max.sum(), 4))
print("  when x_9 is the new runner-up (p2 = 9):", np.round(n_new_ru / n_new_ru.sum(), 4))
print(f"\nt = 9, S > 0.2, p1 < 9, p2 < 9 (n = {len(d)}), by the number of comparators on:")
print("n_on  share   P(s[p2]>s[p1])  P(m[p2]<m[p1])  P(V[p2]>V[p1])  median dV/M  dm/M   ds/M   P(answer = p2)  median x9/S")
for k in range(5):
    c = non == k
    if c.sum() < 50:
        continue
    print(f"{k}     {c.mean():.4f}  {np.mean(ds[c] > 0):.4f}          {np.mean(dm[c] < 0):.4f}          {np.mean(dV[c] > 0):.4f}"
          f"          {np.median(dV[c]):6.1f}    {np.median(dm[c]):6.1f}  {np.median(ds[c]):6.1f}   {corr[c].mean():.4f}          {np.median(xr[c]):.3f}")
c = non < 4
print(f"<4    {c.mean():.4f}  {np.mean(ds[c] > 0):.4f}          {np.mean(dm[c] < 0):.4f}          {np.mean(dV[c] > 0):.4f}"
      f"          {np.median(dV[c]):6.1f}    {np.median(dm[c]):6.1f}  {np.median(ds[c]):6.1f}   {corr[c].mean():.4f}          {np.median(xr[c]):.3f}")
print("\nsame, restricted to gap/M > 0.05:")
for nm, c in (("4", (non == 4) & (gap > 0.05)), ("<4", (non < 4) & (gap > 0.05))):
    print(f"{nm:3s}  n = {c.sum():7d}  P(s[p2]>s[p1]) = {np.mean(ds[c] > 0):.4f}   P(m[p2]<m[p1]) = {np.mean(dm[c] < 0):.4f}   P(answer = p2) = {corr[c].mean():.4f}")
