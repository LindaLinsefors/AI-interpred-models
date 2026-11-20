"""E10. Is the readout-visible part of the state a shift register ('tape')?

Under free resting flow h -> A h (ordinary units), the logits a state WOULD give if read out now are
l = Uo h.  If the tape picture holds there is a 10x10 matrix Sig with  Uo A ~ Sig Uo  and
Sig ~ 'shift every position by -1' (an event that is one step older belongs to the same absolute
position, which is index 9 - age).
"""
from common import *
Uo = U[:, ORD]
A = W[np.ix_(ORD, ORD)]
Sig = Uo @ A @ np.linalg.pinv(Uo)
res = np.linalg.norm(Uo @ A - Sig @ Uo) / np.linalg.norm(Uo @ A)
print("relative residual of  Uo A = Sig Uo :", res)
UA, E = Uo @ A, Uo @ A - Sig @ Uo                       # E: what the 10x10 model misses, row by row
print(f"   split by rows: rows 0-8 together {np.linalg.norm(E[:9]) / np.linalg.norm(UA[:9]):.3f},  row 9 alone {np.linalg.norm(E[9]) / np.linalg.norm(UA[9]):.3f}"
      f"  (row 9 holds {(E[9] ** 2).sum() / (E ** 2).sum():.0%} of the squared residual)")
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("Sig (row = logit index after one free step, column = logit index before):")
print("      " + "".join(f"{j:7d}" for j in range(10)))
for i in range(10):
    print(f"{i:4d}  " + "".join(f"{Sig[i,j]:7.2f}" for j in range(10)))

# The raw superdiagonal Sig[i, i+1] (nothing removed): how much of logit index i+1 lands on index i after one free step
print("\nentries Sig[i, i+1] (shift by -1):", np.round([Sig[i, i+1] for i in range(9)], 2))
print("diagonal Sig[i, i]:", np.round(np.diag(Sig), 2))
off = np.abs(Sig[:7]); off[np.arange(7), np.arange(1, 8)] = 0          # rows 0-6 without their superdiagonal entry
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"largest other |entry| in rows 0-6: {off.max():.3f} (Sig[{i},{j}])")

# Test on real states: logits now vs logits one free step later, on states from real trajectories
x = sample(100_000, 1010)
H = run(x)
for t in (3, 5, 7):
    h = H[:, t][:, ORD]
    l0 = h @ Uo.T
    l1 = (h @ A.T) @ Uo.T
    pred = l0 @ Sig.T
    c0 = l1 - l1[:, :9].mean(1, keepdims=True)
    cp = pred - pred[:, :9].mean(1, keepdims=True)
    r2 = 1 - ((c0[:, :9] - cp[:, :9]) ** 2).sum() / (c0[:, :9] ** 2).sum()
    print(f"t={t}: R2 of predicted free-step logits (positions 0-8, mean removed) = {r2:.4f}")
