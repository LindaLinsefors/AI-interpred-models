"""I4 (max vs runner-up, data-free counterpart). Two-value inputs: M = 1 at p_e, a later input 1 + eps at p_l, background b.
Record offset eps*(p_e, p_l): the later input is treated as the new max iff eps > eps*.  Compared with the medians of
the N(0,1) sweeps (i04b_MR.npz)."""
from icommon import *
import i04b_takeover as tk

Z = np.load(HIDE / "r1_I_errors" / "i04b_MR.npz")
eps, pe, pl = Z["eps"], Z["pe"], Z["pl"]
fin = np.isfinite(eps)
rows = []; pairs = []
tab = {}
for bg in (-1.0, -0.3):
    for a in range(9):
        for b in range(a + 1, 10):
            xx = np.full((1, 10), bg); xx[0, a] = 1.0
            D, _ = tk.sweep(xx, np.array([a]), np.array([b]), "MR", np.array([1.0]))
            tab[(bg, a, b)] = tk.crossing(D)[0][0]
print("two-value record offset eps*(p_e, p_l), background -1 (rows p_e, columns p_l); in brackets the median of the N(0,1) sweeps")
out = []
for a in range(9):
    row = [a]
    for b in range(1, 10):
        if b <= a: row.append("."); continue
        s = fin & (pe == a) & (pl == b)
        med = np.median(eps[s]) if s.sum() >= 50 else np.nan
        row.append(f"{tab[(-1.0, a, b)]:+.3f} [{med:+.3f}]")
        if np.isfinite(med): pairs.append((tab[(-1.0, a, b)], tab[(-0.3, a, b)], med))
    out.append(row)
print(fmt_table(out, ["p_e \ p_l"] + [str(b) for b in range(1, 10)]))
P = np.array(pairs)
print(f"over {len(P)} cells: corr(two-value b=-1, N(0,1) median) = {np.corrcoef(P[:, 0], P[:, 2])[0, 1]:.3f}; corr(two-value b=-0.3, N(0,1) median) = {np.corrcoef(P[:, 1], P[:, 2])[0, 1]:.3f}; "
      f"corr(b=-1, b=-0.3) = {np.corrcoef(P[:, 0], P[:, 1])[0, 1]:.3f}")
print(f"two-value |eps*| (b = -1): mean {np.abs(P[:, 0]).mean():.4f}, max {np.abs(P[:, 0]).max():.4f};  share of the 45 cells with eps* > 0: {np.mean(P[:, 0] > 0):.2f}")
