"""E25. A few exact numbers quoted in the report, computed from the weights in one place."""
from common import *
from modes import LEFT, RIGHT, LAM, NAMES

g = -w[COMP]
print("input weights:", {u: round(float(w[u]), 3) for u in range(16)})
print("\nreadout of the current input: U[9, COMP] * w[COMP] =", np.round(U[9, COMP] * w[COMP], 2), " sum =", round(float(U[9, COMP] @ w[COMP]), 2))
print("U[9,4] * w[4] =", round(float(U[9, 4] * w[4]), 2), "   U[8,5] =", round(float(U[8, 5]), 3), "  U[8,5]*18.01 =", round(float(U[8, 5] * 18.009), 2))
print("what one unit of input sends to units 5 and 9 one step later (all 13 non-pulse units on: W[u, ORD] @ w[ORD]): ",
      np.round(W[np.ix_([5, 9], ORD)] @ w[ORD], 3))
print("  split for unit 5: comparators", np.round(W[5, COMP] * w[COMP], 2), " wave units total", round(float(W[5, WAVE] @ w[WAVE]), 3))
print("W[5,4]*w4 =", round(float(W[5, 4] * w[4]), 2), "  W[9,4]*w4 =", round(float(W[9, 4] * w[4]), 2), "  W[2,4]*w4 =", round(float(W[2, 4] * w[4]), 2))
print("unit 2 self weight", round(float(W[2, 2]), 4), "  W[2,4] =", round(float(W[2, 4]), 4), "  W[4,2] =", round(float(W[4, 2]), 4), "  W[4,4] =", round(float(W[4, 4]), 4))
print("rows of the comparator block sum (Bc 1):", np.round((W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]).sum(1), 4))
print("columns: what the input direction leaves in every unit one step later, W[:, ORD] @ w[ORD]:")
v = W[:, ORD] @ w[ORD]
print("   ", {u: round(float(v[u]), 3) for u in range(16)})
print("cos(angle) between each unit's row on the comparators and the comparators' input weights:")
for u in range(16):
    r = W[u, COMP]
    print(f"    unit {u:2d}: {float(r @ w[COMP] / np.linalg.norm(r) / np.linalg.norm(w[COMP])):+.3f}   |row| = {np.linalg.norm(r):.2f}")
print("\nreadout columns of the wave units with their most negative entry (position):")
for u in WAVE:
    print(f"   unit {u:2d}: min U = {U[:, u].min():6.2f} at position {int(U[:, u].argmin())}   max U = {U[:, u].max():5.2f} at position {int(U[:, u].argmax())}")
print("\nreadout columns of the comparators (positions 0..9):")
for u in COMP:
    print(f"   unit {u}: ", np.round(U[:, u], 2))
# sinusoid fit of comparator readout columns over positions 0..7
om = np.angle(LAM["clock"])
for u in COMP:
    p = np.arange(8)
    X = np.column_stack([np.cos(om * p), np.sin(om * p), np.ones(8)])
    c, *_ = np.linalg.lstsq(X, U[:8, u], rcond=None)
    fit = X @ c
    r2 = 1 - ((U[:8, u] - fit) ** 2).sum() / ((U[:8, u] - U[:8, u].mean()) ** 2).sum()
    print(f"   unit {u}: U[p] ~ {np.hypot(c[0], c[1]):.2f} cos(42.75 p - {np.degrees(np.arctan2(c[1], c[0])):.0f} deg) + {c[2]:.2f},  R2 = {r2:.3f}")
