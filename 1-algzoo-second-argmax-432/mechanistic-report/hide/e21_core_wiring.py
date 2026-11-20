"""E21. Wiring of the 'core' (units 2, 4 and the comparators 1, 6, 7, 8), from the weights.

Thresholds: theta_j(t+1) = -(W[j] . h_t) / w_j  for j in COMP (input units).  Writing the comparator
activations as h_k = g_k (theta_k(t) - x_t)  (g_k = |w_k|, when comparator k is on) gives

   theta(t+1) = Bc (theta(t) - x_t 1) + b2 h_2(t) + b4 h_4(t) + (other units)

with Bc[j,k] = W[j,k] g_k / g_j,  b2[j] = W[j,2]/g_j,  b4[j] = W[j,4]/g_j.
"""
from common import *

g = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
print("comparators:", COMP, " gains g =", np.round(g, 3))
print("\nBc (thresholds -> next thresholds, in input units):")
for j, row in zip(COMP, Bc):
    print(f"  theta_{j}' <- " + "  ".join(f"{v:7.3f}*theta_{k}" for v, k in zip(row, COMP)))
print("\nBc @ (1,1,1,1) =", np.round(Bc @ np.ones(4), 4), "  <- what a common shift of all four thresholds (or the input x) leaves")
lam, V = np.linalg.eig(Bc)
print("eigenvalues of Bc:", ", ".join(f"{abs(l):.4f}@{np.degrees(np.angle(l)):.2f}deg" for l in lam))
k = int(np.argmax(lam.imag))
v = V[:, k]
print("rotating eigenvector (threshold swing per comparator, amplitude@phase):",
      ", ".join(f"theta_{j}: {abs(c)/abs(v).max():.2f}@{np.degrees(np.angle(c/v[COMP.index(7)])):.0f}" for j, c in zip(COMP, v)))

b2 = W[COMP, 2] / g
b4 = W[COMP, 4] / g
print("\nlevel supply from unit 2:  b2 =", np.round(b2, 4), "  (= -W[j,2]/w_j)")
print("record supply from unit 4: b4 =", np.round(b4, 4), "  times h4 = 10.157*d gives", np.round(b4 * 10.157, 3), "per unit increment d")
print("unit 4's own threshold: theta_4' = (", " + ".join(f"{-W[4,u]/w[4]:.4f} h_{u}" for u in (2, 4, 1, 6, 7, 8)), "+ ... )")
print("unit 2: h_2' =", " + ".join(f"{W[2,u]:.3f} h_{u}" for u in (2, 4, 1, 6, 7, 8, 9, 12)), "+ ...   input weight", round(w[2], 4))

# ---- measured decomposition of each threshold on real data
x = sample(400_000, 2121)
H = run(x)
st = running_stats(x)
M, S, a1 = st["M1"], st["M2"], st["a1"]
groups = {"unit 2": [2], "comparators": COMP, "unit 4": [4], "units 5,9": [5, 9], "wave units (not 2)": [0, 3, 10, 11, 12, 13, 14, 15]}
print("\nmeasured parts of theta_j(t+1) / M_t at steps t with max-age >= 2 and M > 0.3 (mean over steps; then sd of each part):")
sel = (a1[:, :9] >= 2) & (M[:, :9] > 0.3)
Hs = H[:, :9][sel]; Ms = M[:, :9][sel]
for j in COMP + [4]:
    row = []
    for name, us in groups.items():
        part = -(Hs[:, us] @ W[j, us]) / w[j] / Ms
        row.append(f"{name}: {part.mean():6.3f} (sd {part.std():.3f})")
    tot = -(Hs @ W[j]) / w[j] / Ms
    print(f"  theta_{j}: " + " | ".join(row) + f" | total {tot.mean():.3f} (sd {tot.std():.3f})")
print("\nunit 2 itself: h_2 / (17.3 M_{t-1}) by age of the max at t (median):",
      [round(float(np.median((H[:, 1:, 2] / (17.3 * M[:, :-1]))[(a1[:, 1:] == a) & (M[:, :-1] > 0.3)])), 3) for a in range(1, 9)])
