"""Reviewer R2, test 2b.  Which part of the satellites' feedback into the six core units carries the accuracy?
Full network, only the listed weights W[core, source] are changed.
python -B t2b_core_feedback_parts.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import RIGHT, LEFT, NAMES, IS_COMPLEX

core = [2, 4, 1, 6, 7, 8]
comp_and_4 = [4, 1, 6, 7, 8]
x = sample(600_000, 9203)
truth = second_argmax(x)
base = logits(run(x)).argmax(1)
print(f"baseline accuracy {np.mean(base == truth):.4f}")

def acc(W2, label):
    a = logits(run(x, W=W2)).argmax(1)
    print(f"  {label:70s}: accuracy {np.mean(a == truth):.4f}   agreement {np.mean(a == base):.4f}")

for label, rows, cols in (
        ("zero unit 5 -> core", core, [5]),
        ("zero unit 9 -> core", core, [9]),
        ("zero units 5, 9 -> core", core, [5, 9]),
        ("zero wave units (not 2) -> core", core, [0, 3, 10, 11, 12, 13, 14, 15]),
        ("zero wave units (not 2) -> unit 2 only", [2], [0, 3, 10, 11, 12, 13, 14, 15]),
        ("zero wave units (not 2) -> unit 4 only", [4], [0, 3, 10, 11, 12, 13, 14, 15]),
        ("zero wave units (not 2) -> comparators only", COMP, [0, 3, 10, 11, 12, 13, 14, 15]),
        ("zero all satellites -> comparators only", COMP, [0, 3, 5, 9, 10, 11, 12, 13, 14, 15]),
        ("zero all satellites -> units 2 and 4 only", [2, 4], [0, 3, 5, 9, 10, 11, 12, 13, 14, 15])):
    W2 = W.copy(); W2[np.ix_(rows, cols)] = 0.0
    acc(W2, label)

# keep only selected eigenmode components of the wave units' feedback into the core:
# replace W[core, wave8] h[wave8] by W[core, wave8] P h[ORD], P = projector on chosen modes restricted to wave8 rows
print("\nwave units' feedback into the core restricted to chosen eigenmodes of the resting map (the feedback then reads")
print("W[core, wave8] (sum over chosen modes of RIGHT_k LEFT_k) h[ORD]; pulses 5, 9 -> core kept):")
oi = np.array(ORD)
wave8 = [0, 3, 10, 11, 12, 13, 14, 15]
w8i = [ORD.index(u) for u in wave8]

def proj(names):
    P = np.zeros((13, 13))
    for n in names:
        if IS_COMPLEX[n]:
            P += 2 * np.real(np.outer(RIGHT[n], LEFT[n]))
        else:
            P += np.real(np.outer(RIGHT[n], LEFT[n]))
    return P

def run_proj(names):
    P = proj(names)
    F = W[np.ix_(core, wave8)] @ P[w8i]                 # (6, 13): feedback as a function of h[ORD]
    Wz = W.copy(); Wz[np.ix_(core, wave8)] = 0.0
    B = len(x); h = np.zeros((B, N))
    for t in range(T):
        z = x[:, t, None] * w + h @ Wz.T
        z[:, core] += h[:, oi] @ F.T
        h = np.maximum(z, 0)
    a = (h @ U.T).argmax(1)
    print(f"  modes kept {str(names):60s}: accuracy {np.mean(a == truth):.4f}   agreement {np.mean(a == base):.4f}")

run_proj(NAMES)                                         # sanity: all modes = identity -> baseline
run_proj(["level"])
run_proj(["level", "runner"])
run_proj(["level", "runner", "clock"])
run_proj(["level", "runner", "clock", "short"])
run_proj(["level", "runner", "clock", "short", "trans", "echo"])
run_proj(["level", "runner", "clock", "h2", "alt", "h3"])
run_proj(["h2", "alt", "h3", "short", "trans", "echo"])
