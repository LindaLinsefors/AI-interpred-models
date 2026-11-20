"""Reviewer R2, test 8.  Section 4.1: "The level, runner and clock modes account for 96.4-99.6% of the variance of each
threshold."  Substitution test: run the full network, but let the five threshold units (comparators 1, 6, 7, 8 and
unit 4) read the 13 ordinary units only through the chosen eigenmodes (W[rows, ORD] -> W[rows, ORD] P, P = projector
on the chosen modes).  The pulses' direct weights into these units are kept.  Everything else is untouched.
python -B t8_threshold_substitution.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, NAMES, IS_COMPLEX

x = sample(600_000, 9911)
truth = second_argmax(x)
H0, Z0 = run(x, return_pre=True)
base = logits(H0).argmax(1)
oi = np.array(ORD)
rows = COMP + [4]

def proj(names):
    P = np.zeros((13, 13))
    for n in names:
        P += (2 * np.real(np.outer(RIGHT[n], LEFT[n])) if IS_COMPLEX[n] else np.real(np.outer(RIGHT[n], LEFT[n])))
    return P

print(f"baseline accuracy {np.mean(base == truth):.4f}")
print("modes read by the five threshold units | accuracy | agreement | R2 of the five thresholds' recurrent drive (vs the real run) | unit 4: fires on non-records / misses records")
st = running_stats(x)
newmax = x[:, 1:] > np.maximum(st["M1"][:, :-1], 0)
for names in (NAMES, ["level", "runner", "clock"], ["level", "runner", "clock", "short"], ["level", "runner", "clock", "trans"],
              ["level", "runner", "clock", "short", "trans"], ["level", "runner", "clock", "short", "trans", "echo"],
              ["level", "runner", "clock", "h2", "alt", "h3"], ["level", "runner", "clock", "short", "trans", "echo", "h2"],
              ["level", "runner", "clock", "short", "trans", "echo", "h2", "h3"]):
    W2 = W.copy()
    W2[np.ix_(rows, ORD)] = W[np.ix_(rows, ORD)] @ proj(names)
    H, Z = run(x, W=W2, return_pre=True)
    a = logits(H).argmax(1)
    # variance of the recurrent drive explained on the REAL trajectory (the report's kind of number)
    d_real = H0[:, :-1][:, :, oi] @ W[np.ix_(rows, ORD)].T
    d_mode = H0[:, :-1][:, :, oi] @ W2[np.ix_(rows, ORD)].T
    r2 = 1 - ((d_real - d_mode) ** 2).sum((0, 1)) / ((d_real - d_real.mean((0, 1))) ** 2).sum((0, 1))
    f = Z[:, 1:, 4] > 0
    print(f"  {str(names):78s} | {np.mean(a == truth):.4f} | {np.mean(a == base):.4f} | " + " ".join(f"{v:.3f}" for v in r2)
          + f" | {f[~newmax].mean():.4f} / {1 - f[newmax].mean():.4f}")
