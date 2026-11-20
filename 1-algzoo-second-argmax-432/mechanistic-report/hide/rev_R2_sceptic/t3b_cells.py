"""Reviewer R2, test 3b.  Full (p_S, p_T) tables of P(answer = third) for a max at position 9, 8, 7
(max arrives after both; T > 0.2; relative gap (S - T)/S in (0.08, 0.2]).  Rows: p_S, columns: p_T.
python -B t3b_cells.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

rows = []
for seed in range(24):
    x = sample(500_000, 9400 + seed)
    ans = logits(run(x)).argmax(1)
    order = np.argsort(-x, axis=1)
    pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
    xs = np.take_along_axis(x, order[:, :3], 1)
    rel = (xs[:, 1] - xs[:, 2]) / xs[:, 1]
    ok = (pM > pS) & (pM > pT) & (xs[:, 2] > 0.2) & (rel > 0.08) & (rel <= 0.2)
    rows.append(np.column_stack([pM, pS, pT, ans == pT, ans == pS, ans == pM])[ok])
R = np.concatenate(rows).astype(int)
pM, pS, pT, e3, okk, eM = R.T
print("n selected:", len(R), "  overall P(answer = third) =", round(e3.mean(), 4))
for pm in (9, 8, 7, 6):
    print(f"\nmax at position {pm}:  P(answer = third) [rows p_S = runner-up, columns p_T = third];  '.' = fewer than 150 cases")
    print("  pS\\pT " + "".join(f"{c:7d}" for c in range(pm)))
    for r in range(pm):
        line = f"   {r}    "
        for c in range(pm):
            q = (pM == pm) & (pS == r) & (pT == c)
            line += f"{e3[q].mean():7.3f}" if q.sum() >= 150 else "      ."
        print(line)
    q = pM == pm
    print(f"  (all cells: P(third) = {e3[q].mean():.4f}, P(runner-up) = {okk[q].mean():.4f}, P(max) = {eM[q].mean():.4f}, n = {q.sum()})")
