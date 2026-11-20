"""Reviewer R2, test 3.  Report section 7 / section 11 item 7: "errors at separations 7 and 8 as a consequence
of the single-harmonic deposit".  Separations D = |p_S - p_T| of 7 or 8 can only occur with one candidate at
position 0..2 and the other at 7..8 and the max at 8 or 9, so D is confounded with 'the later candidate and the
max arrive at the very end'.  Same selection as the lead's e22 (max arrives after both S and T, T > 0.2).
P(answer = third) by D, split by (i) the lag between the later candidate and the max, (ii) the position of the
later candidate, (iii) which of the two came first.
python -B t3_separation_vs_end_effect.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

rows = []
for seed in range(12):
    x = sample(500_000, 9300 + seed)
    ans = logits(run(x)).argmax(1)
    order = np.argsort(-x, axis=1)
    pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
    xs = np.take_along_axis(x, order[:, :3], 1)
    ok = (pM > pS) & (pM > pT) & (xs[:, 2] > 0.2)
    rel = (xs[:, 1] - xs[:, 2]) / xs[:, 1]
    rows.append(np.column_stack([pM, pS, pT, ans == pT, ans == pS, rel])[ok])
R = np.concatenate(rows)
pM, pS, pT = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2].astype(int)
e3, okk, rel = R[:, 3] > 0, R[:, 4] > 0, R[:, 5]
D = np.abs(pS - pT); later = np.maximum(pS, pT); lag = pM - later
print("n selected:", len(R))

def cell(q):
    return f"{e3[q].mean():.3f} ({q.sum():6d})" if q.sum() >= 150 else "   -   (      )"

for lo, hi in ((0.03, 0.08), (0.08, 0.2), (0.2, 0.5)):
    g = (rel > lo) & (rel <= hi)
    print(f"\n===== relative gap (S - T)/S in ({lo}, {hi}] :  P(answer = third) (n) =====")
    print("(i) by D (rows) and lag = p_M - later candidate (columns)")
    print("   D  1-cos    all               lag = 1            lag = 2            lag >= 3")
    for d in range(1, 9):
        print(f"   {d}  {1 - np.cos(np.radians(42.75 * d)):.2f}   {cell(g & (D == d))}   {cell(g & (D == d) & (lag == 1))}   {cell(g & (D == d) & (lag == 2))}   {cell(g & (D == d) & (lag >= 3))}")
    print("(ii) by D (rows) and position of the later candidate (columns 1..8)")
    print("   D   " + "".join(f"later={p:<13d}" for p in range(1, 9)))
    for d in range(1, 9):
        print(f"   {d}   " + "  ".join(cell(g & (D == d) & (later == p)) for p in range(1, 9)))
    print("(iii) by D and order: 'T first' = the third was the earlier one (it is an overtaken maximum of the prefix), 'S first' = the third arrived after the runner-up")
    print("   D    T first            S first")
    for d in range(1, 9):
        print(f"   {d}   {cell(g & (D == d) & (pT < pS))}   {cell(g & (D == d) & (pS < pT))}")
    print("(iv) matched comparison: later candidate at position 7, max at 8 or 9; earlier candidate at position 0..6")
    print("   earlier:  " + "".join(f"{p:<19d}" for p in range(0, 7)))
    print("   T first   " + "  ".join(cell(g & (later == 7) & (pT < pS) & (np.minimum(pS, pT) == p)) for p in range(0, 7)))
    print("   S first   " + "  ".join(cell(g & (later == 7) & (pS < pT) & (np.minimum(pS, pT) == p)) for p in range(0, 7)))
    print("(v) matched comparison: later candidate at position 5, max at 6..9; earlier candidate at position 0..4")
    print("   earlier:  " + "".join(f"{p:<19d}" for p in range(0, 5)))
    print("   T first   " + "  ".join(cell(g & (later == 5) & (pT < pS) & (np.minimum(pS, pT) == p)) for p in range(0, 5)))
    print("   S first   " + "  ".join(cell(g & (later == 5) & (pS < pT) & (np.minimum(pS, pT) == p)) for p in range(0, 5)))
