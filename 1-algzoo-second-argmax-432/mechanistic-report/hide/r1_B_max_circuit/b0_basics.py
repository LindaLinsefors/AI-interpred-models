"""B0. Weights that the max circuit uses, and re-verification of the lead's facts F2-F5 that arm B builds on.
No data except one accuracy check (seed 1000).
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

print("w (input weights):")
for u in range(N):
    print(f"  unit {u:2d}: {w[u]:9.4f}")

print("\nW rows (to-unit) for units 4, 5, 9, 2;  columns = from-unit 0..15")
for u in (4, 5, 9, 2):
    print(f"  row {u}: " + " ".join(f"{v:7.3f}" for v in W[u]))
print("\nW columns (from-unit) 4, 5, 9: what the pulses send")
for u in PULSE:
    print(f"  col {u}: " + " ".join(f"{v:7.3f}" for v in W[:, u]))

print("\neigenmodes of A = W[ORD, ORD]")
for n in NAMES:
    print(f"  {n:7s} lam = {LAM[n]:.5f}  |lam| = {abs(LAM[n]):.5f}  angle = {np.degrees(np.angle(LAM[n])):8.3f}")

print("\ngauge check: LEFT . W[ORD, 4] (should be 1 for every mode)")
print("  ", {n: np.round(LEFT[n] @ W[ORD, 4], 6) for n in NAMES})

print("\nF4: cos(w[ORD], RIGHT echo) and A w[ORD]")
wo = w[ORD]
re = RIGHT["echo"].real
print("  cosine:", wo @ re / np.linalg.norm(wo) / np.linalg.norm(re))
print("  A w[ORD]:", A @ wo)
print("  W[PULSE, ORD] w[ORD]:", W[np.ix_(PULSE, ORD)] @ wo)

print("\nmode coordinates of the input direction w[ORD] (LEFT . w[ORD]) and of the pulse columns")
print(f"  {'mode':7s} {'w[ORD]':>22s} {'W[ORD,4]':>22s} {'W[ORD,5]':>22s} {'W[ORD,9]':>22s}")
for n in NAMES:
    vals = [LEFT[n] @ wo, LEFT[n] @ W[ORD, 4], LEFT[n] @ W[ORD, 5], LEFT[n] @ W[ORD, 9]]
    if IS_COMPLEX[n]:
        print(f"  {n:7s} " + " ".join(f"{abs(v):12.4f}@{np.degrees(np.angle(v)):7.1f}d " for v in vals))
    else:
        print(f"  {n:7s} " + " ".join(f"{v.real:22.4f}" for v in vals))

print("\nLEFT level and RIGHT level per ordinary unit (real)")
Ll, Rl = LEFT["level"].real, RIGHT["level"].real
print(f"  {'unit':>4s} {'LEFT':>9s} {'RIGHT':>9s} {'L*R':>9s} {'|L|^2 share':>12s}")
for i, u in enumerate(ORD):
    print(f"  {u:4d} {Ll[i]:9.4f} {Rl[i]:9.4f} {Ll[i]*Rl[i]:9.4f} {Ll[i]**2/np.sum(Ll**2):12.4f}")
print("  max imaginary part of LEFT/RIGHT level:", abs(LEFT["level"].imag).max(), abs(RIGHT["level"].imag).max())
print("  sum L*R =", Ll @ Rl)

x = sample(400_000, 1000)
print("\naccuracy N(0,1), seed 1000, n=400k:", accuracy(x))
x = sample(400_000, 1001, "uniform")
print("accuracy Unif(-1,1), seed 1001, n=400k:", accuracy(x))
