"""S00. Print the weights I build on (nothing is computed from data here)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=2, suppress=True, linewidth=250)
print("w (input weights), units 0..15:\n", w)
print("\nW (rows = receiving unit, cols = sending unit):")
print("        " + "".join(f"{j:7d}" for j in range(16)))
for i in range(16):
    print(f"to {i:2d}: " + "".join(f"{v:7.2f}" for v in W[i]))
print("\nU (rows = position, cols = unit):")
print("        " + "".join(f"{j:7d}" for j in range(16)))
for p in range(10):
    print(f"pos {p}: " + "".join(f"{v:7.2f}" for v in U[p]))
print("\ncolumn sums of U:", U.sum(0))
print("\nU[:,ORD] w[ORD] (logits per unit raw input at age 0):", U[:, ORD] @ w[ORD])
print("W[PULSE, ORD] w[ORD]:", W[np.ix_(PULSE, ORD)] @ w[ORD])
print("A w[ORD]:", A @ w[ORD])
for n in NAMES:
    print(f"{n:7s} |lam| = {abs(LAM[n]):.4f} angle = {np.degrees(np.angle(LAM[n])):7.2f}   LEFT.w[ORD] = {LEFT[n] @ w[ORD]:.3f}")
