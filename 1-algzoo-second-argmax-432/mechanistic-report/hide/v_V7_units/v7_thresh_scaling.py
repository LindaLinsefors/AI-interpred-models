"""V7 attack 5 (deterministic): do the comparator THRESHOLDS oscillate with the age of the max with an amplitude
proportional to the gap M - S?

Controlled inputs: background b everywhere, M = 1.5 at p_M, S at p_S; thresholds for the next step p_v:
theta_j = -(W h_{p_v - 1})_j / w_j  (the input value at which unit j switches).  All (p_S, p_M) below p_v.
If the oscillation were exactly proportional to the gap, the half-range over ages of theta_j / M would be
(1 - S/M) x its value at S = 0, and f_j = (theta_j - S)/(M - S) would not depend on S.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

M0 = 1.5
UN = [1, 6, 7, 8, 4]
RATIOS = [0.0, 0.1, 0.25, 1 / 3, 0.5, 0.75, 0.9, 0.97, 0.995]


def thetas(pv, bg, S):
    """theta[a1, a2, unit] / M for all positions of max and runner-up below pv (nan where a1 == a2)."""
    out = np.full((pv + 1, pv + 1, 5), np.nan)
    for pM in range(pv):
        for pS in range(pv):
            if pM == pS:
                continue
            x = np.full(pv, bg, dtype=float)
            x[pM], x[pS] = M0, S
            h = run(x)[0, -1]
            out[pv - pM, pv - pS] = (-(W @ h) / w)[UN] / M0
    return out


for bg in (-1.0, 0.0):
    for pv in (9, 6):
        print(f"\n================ background {bg}, thresholds for step p_v = {pv}; ages of the max 1..{pv} ================")
        TH = {r: thetas(pv, bg, r * M0) for r in RATIOS}
        print("  theta_j / M by age of the max (mean over the runner-up's age), for S/M = 1/3 (the lead's sweep ratio) and S/M = 0.9:")
        for r in (1 / 3, 0.9):
            mu = np.nanmean(TH[r][1:, 1:], axis=1)                    # (ages, units)
            for i, j in enumerate(UN):
                print(f"    S/M = {r:.2f}  unit {j}: " + " ".join(f"{v:6.3f}" for v in mu[:, i]))
        print("\n  half-range over the ages of the max of theta_j / M (mean over runner-up ages first):")
        print("    S/M    1-S/M  " + "".join(f"   unit {j}" for j in UN) + "   |  relative to S/M = 0: " + "".join(f"  unit {j}" for j in UN)
              + "  | mean level theta/M: " + "".join(f"  unit {j}" for j in UN))
        hr0 = None
        for r in RATIOS:
            mu = np.nanmean(TH[r][1:, 1:], axis=1)
            hr = np.ptp(mu, axis=0) / 2
            hr0 = hr if hr0 is None else hr0
            print(f"    {r:5.3f}  {1 - r:5.3f}  " + "".join(f"{v:9.3f}" for v in hr) + "   |                        "
                  + "".join(f"{v:8.3f}" for v in hr / hr0) + "  |                     " + "".join(f"{v:8.3f}" for v in mu.mean(0)))
        print("\n  is f_j = (theta_j - S)/(M - S) independent of S?  f_j by age of the max (mean over runner-up ages):")
        for i, j in enumerate(UN):
            print(f"    unit {j}:")
            for r in (0.0, 1 / 3, 0.75, 0.9, 0.97):
                f = (np.nanmean(TH[r][1:, 1:, i], axis=1) - r) / (1 - r)
                print(f"       S/M = {r:5.3f}: " + " ".join(f"{v:6.2f}" for v in f))
        # spread over the runner-up's age at fixed age of the max
        r = 1 / 3
        sd = np.nanstd(TH[r][1:, 1:], axis=1).mean(0)
        print(f"\n  spread of theta_j/M over the runner-up's age at fixed age of the max (S/M = 1/3), mean sd: " + "  ".join(f"unit {j}: {v:.3f}" for j, v in zip(UN, sd)))
