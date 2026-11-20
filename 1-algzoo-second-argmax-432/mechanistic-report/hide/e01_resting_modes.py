"""E01. Eigen-structure of the 'resting' linear map (pulses 4, 5, 9 off, the other 13 units on).

Checks, from the weights alone:
  * eigenvalues of A = W[ORD, ORD];
  * is the level eigenvector (lambda ~ 1.01) positive on all 13 ordinary units?  (ersatz bias)
  * is the input direction w[ORD] (almost) the eigenvector with eigenvalue ~ 0?  (low inputs leave no trace)
  * how do the pulse units read each mode (W[PULSE, ORD] @ right eigenvectors)?
"""
from common import *

A = W[np.ix_(ORD, ORD)]
lam, V = np.linalg.eig(A)                  # right eigenvectors: A V = V diag(lam)
order = np.argsort(-np.abs(lam))
lam, V = lam[order], V[:, order]
Linv = np.linalg.inv(V)                    # rows = left eigenvectors (dual basis): Linv @ V = I

print("eigenvalues of the resting map (13 ordinary units):")
for k, l in enumerate(lam):
    print(f"  {k:2d}  |lam| = {abs(l):.4f}   angle = {np.degrees(np.angle(l)):8.2f} deg   lam = {l:.4f}")

# ---- level mode
k_level = int(np.argmin(np.abs(lam - 1.0098)))
v_level = np.real(V[:, k_level])
v_level = v_level * np.sign(v_level.sum())
print("\nlevel eigenvalue:", lam[k_level].real)
print("level right eigenvector on ordinary units (normalised so unit 2 = 1):")
for u, val in zip(ORD, v_level / v_level[ORD.index(2)]):
    print(f"  unit {u:2d}: {val:7.3f}")
print("all positive:", bool((v_level > 0).all()))
print("pulse pre-activations under the level vector (W[PULSE, ORD] @ v, same normalisation):",
      (W[np.ix_(PULSE, ORD)] @ v_level) / v_level[ORD.index(2)])

# ---- runner mode
k_run = int(np.argmin(np.abs(lam - 0.9736)))
v_run = np.real(V[:, k_run])
print("\nrunner eigenvalue:", lam[k_run].real)
print("runner right eigenvector (normalised to max |entry| = 1):")
for u, val in zip(ORD, v_run / np.abs(v_run).max()):
    print(f"  unit {u:2d}: {val:7.3f}")

# ---- echo mode vs input direction
k_echo = int(np.argmin(np.abs(lam)))
v_echo = np.real(V[:, k_echo])
w_ord = w[ORD]
cosang = v_echo @ w_ord / np.linalg.norm(v_echo) / np.linalg.norm(w_ord)
print("\necho eigenvalue:", lam[k_echo].real, "  cos(angle between echo eigenvector and w[ORD]) =", cosang)
print("A @ w[ORD]  (what a unit input leaves in the ordinary units one step later, if all 13 stay on):")
for u, val in zip(ORD, A @ w_ord):
    print(f"  unit {u:2d}: {val:7.3f}")
print("W[PULSE, ORD] @ w[ORD] (what it sends to pulses 4, 5, 9):", W[np.ix_(PULSE, ORD)] @ w_ord)
print("mode coordinates of the input direction  Linv @ w[ORD]  (|.|):")
for k, c in enumerate(Linv @ w_ord):
    print(f"  mode {k:2d} (|lam|={abs(lam[k]):.3f}, {np.degrees(np.angle(lam[k])):7.2f} deg): |coef| = {abs(c):.4f}")

np.savez(HIDE / "e01_modes.npz", lam=lam, V=V, Linv=Linv)
