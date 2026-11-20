"""Reviewer R2, test 6.
(A) Section 5.4 / section 11 item 4: "unit 5's pulse after a maximum prepares that maximum for being overtaken".
    Alternative: unit 5's pulse is the ordinary record that EVERY candidate needs.  Remove the single unit-5 pulse
    one step after the arrival of the final runner-up, for the two orders:
      RM: the runner-up is an overtaken maximum (arm F's case),   MR: the runner-up arrived after the maximum.
    Also: remove the pulse one step after the arrival of the FINAL THIRD (control: should not matter much).
(B) How much of the 'overtaken maximum' case is covered by the clock deposit?  Share of RM sequences by the age g
    of the old max at the overtaking (g = 1: old max arrived the step before), and the effect of erasing the clock
    one step before the overtaking, by g.
python -B t6_pulse5_and_g1.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT

x = sample(1_500_000, 9801)
order = np.argsort(-x, axis=1)
pM, pS, pT = order[:, 0], order[:, 1], order[:, 2]
xs = np.take_along_axis(x, order[:, :3], 1)
Mv, Sv, Tv = xs[:, 0], xs[:, 1], xs[:, 2]
base = logits(run(x)).argmax(1)
oi = np.array(ORD)

def run_clamp(xq, step, unit):
    """set unit to 0 at the given step (per sequence); step = -1 means no clamp."""
    B = len(xq); h = np.zeros((B, N))
    for t in range(T):
        h = np.maximum(xq[:, t, None] * w + h @ W.T, 0)
        sel = np.where(step == t)[0]
        h[sel, unit] = 0.0
    return (h @ U.T).argmax(1)

good = (Sv > 0.2) & (Mv - Sv > 0.1 * Mv) & (Sv - Tv > 0.1 * Mv)
print("(A) one unit-5 pulse removed (the step after the named input arrived).  S > 0.2, both gaps > 0.1 M.")
print("    case | n | baseline P(answer = runner-up) | pulse after the RUNNER-UP removed: P(runner-up), P(max), P(third) | pulse after the THIRD removed: P(runner-up) | pulse after the MAX removed: P(runner-up), P(max)")
for label, cond in (("RM (runner-up = overtaken max), pS <= pM - 2", (pS <= pM - 2)),
                    ("RM, pS = pM - 1", (pS == pM - 1)),
                    ("MR (runner-up after the max), pS <= 8", (pS > pM) & (pS <= 8))):
    q = good & cond & (pT <= 8) & (pM <= 8)
    xq = x[q]
    a_S = run_clamp(xq, pS[q] + 1, 5)
    a_T = run_clamp(xq, pT[q] + 1, 5)
    a_M = run_clamp(xq, pM[q] + 1, 5)
    print(f"    {label:45s} | {q.sum():6d} | {np.mean(base[q] == pS[q]):.3f} | {np.mean(a_S == pS[q]):.3f}  {np.mean(a_S == pM[q]):.3f}  {np.mean(a_S == pT[q]):.3f} | "
          f"{np.mean(a_T == pS[q]):.3f} | {np.mean(a_M == pS[q]):.3f}  {np.mean(a_M == pM[q]):.3f}")

print("\n(B) runner-up = overtaken max (pS < pM), S > 0.2: share by g = pM - pS, and the clock erased one step before the overtaking")
Lc, Rc = LEFT["clock"], RIGHT["clock"]
def run_noclock(xq, step):
    B = len(xq); h = np.zeros((B, N))
    for t in range(T):
        h = np.maximum(xq[:, t, None] * w + h @ W.T, 0)
        sel = np.where(step == t)[0]
        if len(sel):
            z = h[sel][:, oi] @ Lc
            h[np.ix_(sel, oi)] -= 2 * np.real(np.multiply.outer(z, Rc))
    return (h @ U.T).argmax(1)
rm = (pS < pM) & (Sv > 0.2)
print(f"    RM sequences: {rm.sum()} ({rm.mean():.3f} of all);  share of all sequences whose answer is an overtaken max of age 1: {np.mean(rm & (pM - pS == 1)):.3f}")
print("    g | share of RM | baseline P(answer = old max) | clock erased at step pM - 1: P(old max)  P(new max)  P(old runner-up = final third)")
for g in range(1, 8):
    q = rm & (pM - pS == g)
    a = run_noclock(x[q], pM[q] - 1)
    print(f"    {g} |   {q.sum() / rm.sum():.3f}   |   {np.mean(base[q] == pS[q]):.3f}   |   {np.mean(a == pS[q]):.3f}   {np.mean(a == pM[q]):.3f}   {np.mean(a == pT[q]):.3f}")
