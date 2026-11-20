"""Reviewer N1: weight-only numbers of sections 2 and 3.2/3.3 of report.md.  Run: python -B n1_weights.py"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

print("input weights w:", {i: round(float(w[i]), 3) for i in range(16)})
print("max |w| over wave units except 14:", max(abs(w[i]) for i in WAVE if i != 14))

A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
Sig = Uo @ A @ np.linalg.pinv(Uo)


def show_eigs(name, Mx):
    ev = np.linalg.eigvals(Mx)
    ev = ev[np.argsort(-np.abs(ev))]
    print(name, "eigenvalues (modulus @ angle):")
    for e in ev:
        if e.imag >= -1e-12:
            print(f"   {abs(e):.4f} @ {np.degrees(np.angle(e)):7.2f}")


show_eigs("A", A)
show_eigs("Sigma", Sig)

print("superdiagonal Sigma[i,i+1]:", np.round([Sig[i, i + 1] for i in range(9)], 3))
off = Sig[:7].copy()
for i in range(7):
    off[i, i + 1] = 0
i, j = np.unravel_index(np.abs(off).argmax(), off.shape)
print("largest other |entry| in rows 0-6:", round(abs(off[i, j]), 3), "at", (i, j))
R = Sig @ Uo - Uo @ A
print("relative error rows 0-8:", np.linalg.norm(R[:9]) / np.linalg.norm((Uo @ A)[:9]),
      " row 9:", np.linalg.norm(R[9]) / np.linalg.norm((Uo @ A)[9]),
      " all:", np.linalg.norm(R) / np.linalg.norm(Uo @ A))
UA = Uo @ A
cos = [UA[i] @ Uo[i + 1] / np.linalg.norm(UA[i]) / np.linalg.norm(Uo[i + 1]) for i in range(9)]
print("cos(row i of Uo A, row i+1 of Uo), i=0..8:", np.round(cos, 3), " mean over 0..6:", np.mean(cos[:7]))

# slope of the newest score
d = U[:, ORD] @ w[ORD]
print("U[i,ORD].w[ORD], i=0..9:", np.round(d, 2))
print("(U[9]-U[i]).w[ORD], i=0..8:", np.round(d[9] - d[:9], 2))
print("comparators' part of index 9:", np.round(U[9, COMP] * w[COMP], 3), (U[9, COMP] * w[COMP]).sum(),
      " wave part:", (U[9, WAVE] * w[WAVE]).sum())
print("U[9,4] w4 =", U[9, 4] * w[4], "  U[8,5] =", U[8, 5], "  W[5,ORD].w[ORD] =", W[5, ORD] @ w[ORD],
      " product:", U[8, 5] * (W[5, ORD] @ w[ORD]))
# slope of s[9]-s[i] above all thresholds (all comparators off, unit 4 on, wave units on)
dd = U[:, WAVE] @ w[WAVE] + U[:, 4] * w[4]
print("new-max regime: d s[i]/dx (unit 4 on, comparators off):", np.round(dd, 2),
      " (U[9]-U[i]):", np.round(dd[9] - dd[:9], 2))
