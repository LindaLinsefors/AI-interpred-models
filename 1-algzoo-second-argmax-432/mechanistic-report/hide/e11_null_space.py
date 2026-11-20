"""E11. The readout's null space on the ordinary units vs the resting modes.
Uo = U[:, ORD] is 10 x 13, so 3 directions of ordinary-unit space are invisible to the readout.
How much of each mode's right eigenvector lies in that null space?  And: which combination of
logit differences among positions 0..8 (the part that matters for the argmax there) sees each mode?
"""
from common import *
from modes import RIGHT, LEFT, NAMES, LAM, IS_COMPLEX

Uo = U[:, ORD]
u_, s_, vt = np.linalg.svd(Uo)
print("singular values of U[:, ORD]:", np.round(s_, 2))
Nul = vt[10:]                                 # 3 x 13 orthonormal basis of the null space
Pn = Nul.T @ Nul
print("\nfraction of each mode's right eigenvector (Euclidean norm^2) inside the readout's null space,")
print("and the size of what the readout sees, per unit coordinate:")
for n in NAMES:
    vecs = [RIGHT[n].real, RIGHT[n].imag] if IS_COMPLEX[n] else [RIGHT[n].real]
    for lab, v in zip(("re", "im"), vecs):
        frac = (Pn @ v) @ (Pn @ v) / (v @ v)
        seen = Uo @ v
        c = seen - seen[:9].mean()             # what matters among positions 0..8
        print(f"  {n:6s}.{lab if IS_COMPLEX[n] else '  '}: in null space {frac:6.3f}   |U v| = {np.linalg.norm(seen):7.2f}   "
              f"|U v| on positions 0-8 after removing their mean = {np.linalg.norm(c[:9]):7.2f}   |v| = {np.linalg.norm(v):6.2f}")

# Null space expressed in mode coordinates
print("\nnull-space basis vectors in mode coordinates (|coordinate| per mode):")
for k in range(3):
    row = []
    for n in NAMES:
        c = Nul[k] @ LEFT[n]
        row.append(f"{n}:{abs(c):6.3f}")
    print("  ", "  ".join(row))
