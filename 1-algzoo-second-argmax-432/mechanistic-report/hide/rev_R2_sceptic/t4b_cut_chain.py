"""Reviewer R2, test 4b.  Follow-up to t4 (A): the age-locked cuts of units 0, 12, 11, 10 are strongly super-additive.
ReLU removed only on the step where the max of the prefix has the unit's own age (0: 2, 12: 4, 11: 5, 10: 6).
Reported over ALL sequences: accuracy, P(answer = max), and the change of the margin logit[runner-up] - logit[max]
on sequences whose final max is at least 7 steps old (so that all four ages occurred) and S > 0.
python -B t4b_cut_chain.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from itertools import combinations

x = sample(1_000_000, 9601)
truth = second_argmax(x)
st = running_stats(x)
pM = x.argmax(1)
H0, Z0 = run(x, return_pre=True)
L0 = logits(H0); base = L0.argmax(1)
a1 = st["a1"]; posmax = st["M1"] > 0
age_of = {0: 2, 12: 4, 11: 5, 10: 6}
B = len(x); ar = np.arange(B)
old = (pM <= 2) & (st["M2"][:, 9] > 0)
m0 = (L0[ar, truth] - L0[ar, pM])
print(f"baseline: accuracy {np.mean(base == truth):.4f}, P(answer = max) {np.mean(base == pM):.4f}; "
      f"margin runner-up - max on 'old max' sequences (n = {old.sum()}): median {np.median(m0[old]):.1f}, in units of M: {np.median((m0 / st['M1'][:, 9])[old]):.1f}")

def run_linear(mask):
    h = np.zeros((B, N))
    for t in range(T):
        z = x[:, t, None] * w + h @ W.T
        h = np.where(mask[:, t], z, np.maximum(z, 0))
    return h @ U.T

print("units linearised at own age | accuracy | P(answer = max), all sequences | on 'old max' sequences: P(answer = max), median change of margin (in M)")
for r in (1, 2, 3, 4):
    for units in combinations([0, 12, 11, 10], r):
        m = np.zeros((B, T, N), bool)
        for u in units:
            m[:, :, u] = (a1 == age_of[u]) & posmax
        L = run_linear(m); a = L.argmax(1)
        dm = ((L[ar, truth] - L[ar, pM]) - m0) / st["M1"][:, 9]
        print(f"  {str(units):18s} | {np.mean(a == truth):.4f} | {np.mean(a == pM):.4f} | {np.mean(a[old] == pM[old]):.4f}   {np.median(dm[old]):+6.1f}")

print("\ncontrol: four other wave units linearised on the same steps (ages 2, 4, 5, 6 of the max)")
for units in ([3, 13, 14, 15], [2, 3, 13, 14]):
    m = np.zeros((B, T, N), bool)
    for u, a in zip(units, (2, 4, 5, 6)):
        m[:, :, u] = (a1 == a) & posmax
    L = run_linear(m); a = L.argmax(1)
    print(f"  {str(units):18s} | accuracy {np.mean(a == truth):.4f} | P(answer = max) {np.mean(a == pM):.4f}")
m = np.zeros((B, T, N), bool); m[:, :, [3, 13, 14, 15]] = True
L = run_linear(m); a = L.argmax(1)
print(f"  [3, 13, 14, 15] on all steps | accuracy {np.mean(a == truth):.4f} | P(answer = max) {np.mean(a == pM):.4f}")
