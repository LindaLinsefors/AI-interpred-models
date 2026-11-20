"""Code review of e22_separation_errors.py (claim D4).

The code matches the claim.  The interpretation (errors follow 1 - cos(42.75 deg * D)) is confounded:
a large D = |pS - pT| forces one of the two to be very old and the max to arrive very late, and D = 8
forces positions {0, 8} with the max at 9.  Controls with the confounds held fixed:
  (i)   max at the last step (pM = 9) and the earlier of the two at position 0: positions (0, D), D = 1..8
  (ii)  pM = 9 and the later of the two at position 8: positions (8 - D, 8)
  (iii) pM = 9, D fixed, by position of the earlier one
  (iv)  split by which of the two is older (runner-up first / third first)
Same seeds as e22 (2200..2207) plus 8 more for cell sizes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

bins = [(0.0, 0.03), (0.03, 0.08), (0.08, 0.2), (0.2, 0.5)]
rows = []
for seed in range(16):
    x = sample(500_000, 2200 + seed)
    ans = logits(run(x)).argmax(1)
    order = np.argsort(-x, axis=1)
    pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
    xs = np.take_along_axis(x, order[:, :3], 1)
    S, T_ = xs[:, 1], xs[:, 2]
    ok = (pM > pS) & (pM > pT) & (T_ > 0.2)
    rel = (S - T_) / S
    bi = np.full(len(x), -1)
    for i, (lo, hi) in enumerate(bins):
        bi[(rel > lo) & (rel <= hi)] = i
    keep = ok & (bi >= 0)
    rows.append(np.column_stack([pM, pS, pT, bi, ans == pT, ans == pS, np.full(len(x), seed)])[keep])
R = np.concatenate(rows)
pM, pS, pT, bi, e3, c2, seed = R.T
D = np.abs(pS - pT)
first8 = seed < 8


def cell(q):
    n = q.sum()
    return f"{e3[q].mean():.3f} ({n:5d})" if n >= 150 else f"  .   ({n:5d})"


print("replication of e22 (seeds 2200..2207 only): P(answer = third) by D (rows) and gap bin (columns)")
for d in range(1, 9):
    print(f"   D={d}: " + "  ".join(cell(first8 & (D == d) & (bi == b)) for b in range(4)))

print("\n(i) pM = 9 and min(pS, pT) = 0  -> positions (0, D);  16 seeds;  P(answer = third) (n)")
print("      D   1-cos   " + "   ".join(f"({lo:.2f},{hi:.2f}]" for lo, hi in bins))
for d in range(1, 9):
    q = (pM == 9) & (np.minimum(pS, pT) == 0) & (D == d)
    print(f"      {d}   {1 - np.cos(np.radians(42.75 * d)):.2f}   " + "  ".join(cell(q & (bi == b)) for b in range(4)))
print("\n(ii) pM = 9 and max(pS, pT) = 8  -> positions (8 - D, 8)")
for d in range(1, 9):
    q = (pM == 9) & (np.maximum(pS, pT) == 8) & (D == d)
    print(f"      {d}   {1 - np.cos(np.radians(42.75 * d)):.2f}   " + "  ".join(cell(q & (bi == b)) for b in range(4)))
print("\n(ii') pM = 9 and max(pS, pT) = 7  -> positions (7 - D, 7)   [neither is the newest slot before the record]")
for d in range(1, 8):
    q = (pM == 9) & (np.maximum(pS, pT) == 7) & (D == d)
    print(f"      {d}   {1 - np.cos(np.radians(42.75 * d)):.2f}   " + "  ".join(cell(q & (bi == b)) for b in range(4)))
print("\n(ii'') pM <= 7 (at least two steps after the record), all positions")
for d in range(1, 7):
    q = (pM <= 7) & (D == d)
    print(f"      {d}   {1 - np.cos(np.radians(42.75 * d)):.2f}   " + "  ".join(cell(q & (bi == b)) for b in range(4)))

print("\n(iii) gap bin (0.03, 0.08]: P(answer = third) by D (rows) and by the record step pM (columns 2..9); all positions")
print("      D   " + "   ".join(f"pM={p}        " for p in range(2, 10)))
for d in range(1, 9):
    print(f"      {d}   " + "  ".join(cell((bi == 1) & (D == d) & (pM == p)) for p in range(2, 10)))
print("\n      gap bin (0.08, 0.2]:")
for d in range(1, 9):
    print(f"      {d}   " + "  ".join(cell((bi == 2) & (D == d) & (pM == p)) for p in range(2, 10)))

print("\n(iv) by order of the two: runner-up (old max) EARLIER than third (pS < pT) | runner-up LATER (pS > pT);  gap bins (0.03,0.08] and (0.08,0.2]")
for d in range(1, 9):
    print(f"      D={d}:  pS<pT: " + "  ".join(cell((D == d) & (pS < pT) & (bi == b)) for b in (1, 2)) +
          "   |   pS>pT: " + "  ".join(cell((D == d) & (pS > pT) & (bi == b)) for b in (1, 2)))

print("\n(v) gap bin (0.03, 0.08], pM = 9: P(answer = third) by the positions (rows pS, columns pT)")
print("        pT: " + "   ".join(f"{p:11d}" for p in range(9)))
for a in range(9):
    print(f"   pS={a}: " + "  ".join(cell((bi == 1) & (pM == 9) & (pS == a) & (pT == b)) if a != b else "      -      " for b in range(9)))
