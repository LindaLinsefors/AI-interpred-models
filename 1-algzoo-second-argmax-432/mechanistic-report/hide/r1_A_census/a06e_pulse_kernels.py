"""Cross-check of F8 [EXACT]: free-flow logit kernel of one unit of activation of pulse unit P at step s, read at t = 9:
   age 0 (s = 9):  U[:, P]          age m >= 1 (s = 9 - m):  U[:, ORD] A^(m-1) W[ORD, P]
(pulses assumed off after s, all 13 ordinary units on).  Printed: the kernel entries at positions s-2 .. s+1."""
from cc import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for P in (4, 5, 9):
    print(f"\nunit {P}: rows = age m of the pulse at the end (it fired at step s = 9 - m); entries at positions s-2, s-1, s, s+1")
    for m in range(0, 9):
        K = U[:, P] if m == 0 else U[:, ORD] @ np.linalg.matrix_power(A, m - 1) @ W[ORD, P]
        s = 9 - m
        ent = [f"{K[p]:+.2f}" if 0 <= p <= 9 else "    ." for p in (s - 2, s - 1, s, s + 1)]
        print(f"  m={m} s={s}: " + "  ".join(ent) + f"    max entry {K.max():+.2f} at {K.argmax()}, min {K.min():+.2f} at {K.argmin()}")
