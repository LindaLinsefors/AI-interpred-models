"""E24. The clock defined inside the core alone, and the record's launch of it, from the weights.

z_c = l . theta, with l the left eigenvector of Bc (thresholds -> thresholds) for 0.9375 at +42.56 deg,
normalised so that the record supply b4*w4 (thresholds per unit increment) has coordinate 1.
(i)   l . 1  and  l . b2 : does the level supply excite the clock?  (it should not)
(ii)  compare z_c with the 13-unit clock coordinate of modes.py on data;
(iii) z_c at age 1 against the increment d and against the gap."""
from common import *
from modes import coords

g = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
lam, V = np.linalg.eig(Bc)
Linv = np.linalg.inv(V)
k = int(np.argmax(lam.imag))
b4 = W[COMP, 4] / g * w[4]                 # thresholds per unit increment
b2 = W[COMP, 2] / g                        # thresholds per unit of h_2
l = Linv[k] / (Linv[k] @ b4)
print("eigenvalue:", abs(lam[k]), np.degrees(np.angle(lam[k])))
print("left vector l (per threshold 1, 6, 7, 8): ", " ".join(f"{abs(c):.3f}@{np.degrees(np.angle(c)):.0f}" for c in l))
print("|l . 1| =", abs(l @ np.ones(4)), "   |l . b2| * 17.3 (level supply per unit M) =", abs(l @ b2) * 17.3, "   l . b4 = 1 by construction")
print("for scale: |l| * |b4| =", np.linalg.norm(l) * np.linalg.norm(b4))

x = sample(300_000, 2424)
H = run(x)
st = running_stats(x)
M, S, a1 = st["M1"], st["M2"], st["a1"]
# thresholds for the NEXT input from the state at t (comparator part only would be g*(theta - x)); use actual thresholds
theta = -(H @ W[COMP].T) / w[COMP]                     # (B, T, 4): thresholds at t+1
zc = theta @ l
zfull = coords(H)["clock"] / 10.157
sel = (a1 >= 1) & (M > 0.3)
sel[:, 9] = False
a = zc[sel]; b = zfull[sel]
# z_c is read at the next step (thresholds at t+1), so compare with lam * zfull
lamf = 0.9509 * np.exp(1j * np.radians(42.75))
c = np.vdot(lamf * b, a) / np.vdot(lamf * b, lamf * b)
r2 = 1 - np.sum(abs(a - c * lamf * b) ** 2) / np.sum(abs(a - a.mean()) ** 2)       # centred: residual against the variance of z_c
print(f"\ncore clock z_c(t+1) vs lam * (13-unit clock coordinate)(t): complex slope {abs(c):.3f}@{np.degrees(np.angle(c)):.1f} deg, centred R2 = {r2:.4f}")
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
print("\nz_c by age of the max (thresholds read one step later), fit Re(z_c e^{-i phi}) ~ bM M + bS S+  (no intercept; uncentred R2 = 1 - SSres / sum y^2):")
for A in range(1, 9):
    s = sel & (a1 == A)
    ph = np.angle(np.median(zc[s].real / M[s]) + 1j * np.median(zc[s].imag / M[s]))
    y = (zc[s] * np.exp(-1j * ph)).real
    X = np.column_stack([M[s], Sp[s]])
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ co) ** 2).sum() / (y ** 2).sum()                            # uncentred: residual against sum y^2
    print(f"   age {A}: phase {np.degrees(ph):7.1f}   bM = {co[0]:6.3f}  bS = {co[1]:6.3f}   uncentred R2 = {r2:.4f}")
