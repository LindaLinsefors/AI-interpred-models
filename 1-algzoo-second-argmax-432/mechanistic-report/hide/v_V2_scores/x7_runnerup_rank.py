"""Who ranks the runner-up above the THIRD and the rest: m, V, or only their sum?

t = 9, S > 0.1, p1 < 9, p2 < 9, all four comparators on.  Older inputs = indices 0..8.
 - With the max's own index removed: P(argmax m = p2), P(argmax V = p2), P(argmax s = p2), by D = |p1 - p2|.
 - Where is the largest entry of V relative to the max (d = argmax V - p1)?
 - Rank of the runner-up in m (1 = largest among older inputs).

Usage: python -B x7_runnerup_rank.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5700
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 5

n = np.zeros(9); hm = np.zeros(9); hV = np.zeros(9); hs = np.zeros(9); rank_m = np.zeros((9, 10)); rank_s = np.zeros((9, 10))
dmax = np.zeros(17); hm_full = np.zeros(9)
third_m = np.zeros(9); third_s = np.zeros(9)
for x, H, M, S, p1, p2 in batches(nb, seed0):
    h = H[:, 9]
    sel = (h[:, COMP] > 0).all(1) & (S[:, 9] > 0.1) & (p1[:, 9] < 9) & (p2[:, 9] < 9)
    s, V, m = scores(h[sel])
    P1, P2 = p1[sel, 9], p2[sel, 9]
    r = np.arange(len(P1)); D = np.abs(P1 - P2)
    np.add.at(n, D, 1)
    for acc, Y in ((hm, m), (hV, V), (hs, s)):
        Z = Y[:, :9].copy(); Z[r, P1] = -np.inf
        np.add.at(acc, D, Z.argmax(1) == P2)
    np.add.at(dmax, V[:, :9].argmax(1) - P1 + 8, 1)
    np.add.at(rank_m, (D, (m[:, :9] > m[r, P2][:, None]).sum(1) + 1), 1)
    np.add.at(rank_s, (D, (s[:, :9] > s[r, P2][:, None]).sum(1) + 1), 1)

np.set_printoptions(precision=3, suppress=True, linewidth=200)
print(f"seeds {seed0}..{seed0 + nb - 1}; t = 9, all four comparators on, S > 0.1, p1 < 9, p2 < 9; n = {int(n.sum())}")
print("\nolder inputs with the max's index removed: P(the largest remaining entry is the runner-up), by D = |p1 - p2| = 1..8 and overall")
for nm, acc in (("m", hm), ("V", hV), ("s", hs)):
    print(f"  {nm}: " + " ".join(f"{acc[D] / n[D]:.3f}" for D in range(1, 9)) + f"   overall {acc.sum() / n.sum():.4f}")
print("  (chance among 8 remaining inputs = 0.125)")
print("\nrank of the runner-up among the nine older inputs (1 = largest): distribution over ranks 1..5, by D (rows 1..8) and overall")
print("  in m:")
for D in range(1, 9):
    print("   ", D, rank_m[D, 1:6] / n[D])
print("    all", rank_m[:, 1:6].sum(0) / n.sum())
print("  in s:")
print("    all", rank_s[:, 1:6].sum(0) / n.sum())
print("\nposition of the largest entry of V (older inputs) relative to the max, d = argmax V - p1 (d = -8..8):")
print("   ", dmax / dmax.sum())
print("    P(|d| in {3, 4, 5}) =", dmax[[3, 4, 5, 11, 12, 13]].sum() / dmax.sum())
