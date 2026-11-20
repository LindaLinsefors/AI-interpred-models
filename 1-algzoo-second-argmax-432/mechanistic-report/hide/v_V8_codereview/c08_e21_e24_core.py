"""Code review of e21_core_wiring.py and e24_core_clock.py (claims K0, K1, K3, K4, K6).

 (a) K0: the identity theta(t+1) = Bc [theta(t) - x_t 1]_+ + b2 h_2 + b4 h_4 + (other ten units) is not
     checked numerically by any of the lead's scripts.  Check it on data (max abs error).
 (b) K3 / K4 recomputed with an independent implementation of the running max and its age.
 (c) K6: e24 projects z_c on the phase of its median and regresses only that REAL part on (M, S+),
     reporting an UNCENTRED R2.  Add: centred R2, the size of the discarded component perpendicular to
     the median phase, a complex fit z_c ~ alpha M + beta S+ (both R2 conventions), and the spread of
     the phase.
Same samples as e21 (seed 2121) and e24 (seed 2424).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

g = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
b2 = W[COMP, 2] / g
b4 = W[COMP, 4] / g
other = [u for u in range(16) if u not in COMP + [2, 4]]

# ---------------------------------------------------------------- (a) K0
x = sample(400_000, 2121)
H, Z = run(x, return_pre=True)
theta = (Z - x[:, :, None] * w)[:, :, COMP] / g            # theta_j(t) = (W h_{t-1})_j / g_j, t = 0..9   (theta(0) = 0)
lhs = theta[:, 1:]                                         # theta(t+1), t = 0..8
rhs = (np.maximum(theta[:, :-1] - x[:, :-1, None], 0) @ Bc.T + H[:, :-1, 2, None] * b2 + H[:, :-1, 4, None] * b4
       + (H[:, :-1][:, :, other] @ W[np.ix_(COMP, other)].T) / g)
print("(a) K0: max |lhs - rhs| over", lhs.size, "threshold values:", float(np.abs(lhs - rhs).max()), "  (typical |theta|:", round(float(np.abs(lhs).mean()), 3), ")")
print("    and h_k = g_k [theta_k - x_t]_+ for the comparators: max error", float(np.abs(H[:, :, COMP] - g * np.maximum(theta - x[:, :, None], 0)).max()))
th4 = -(Z - x[:, :, None] * w)[:, :, 4] / w[4]
print("    and h_4 = w_4 [x_t - theta_4]_+ : max error", float(np.abs(H[:, :, 4] - w[4] * np.maximum(x - th4, 0)).max()))

# ---------------------------------------------------------------- (b) K3, K4 independent
Mrun = np.maximum.accumulate(x, axis=1)
is_new = np.concatenate([np.ones((len(x), 1), bool), x[:, 1:] > Mrun[:, :-1]], 1)
last_new = np.maximum.accumulate(np.where(is_new, np.arange(T)[None, :], -1), axis=1)
a1 = np.arange(T)[None, :] - last_new
st = running_stats(x)
print("\n(b) independent running max / age agree with running_stats:", bool(np.array_equal(Mrun, st["M1"])), bool(np.array_equal(a1, st["a1"])))
sel = (a1[:, :9] >= 2) & (Mrun[:, :9] > 0.3)
Hs, Ms = H[:, :9][sel], Mrun[:, :9][sel]
print("    K3: mean (sd) of the parts of theta_j(t+1)/M_t;  columns: unit 2 | wave units except 2 | comparators | total")
for j in COMP + [4]:
    row = []
    for us in ([2], [0, 3, 10, 11, 12, 13, 14, 15], COMP, list(range(16))):
        part = -(Hs[:, us] @ W[j, us]) / w[j] / Ms
        row.append(f"{part.mean():6.3f} ({part.std():.3f})")
    print(f"       theta_{j}: " + " | ".join(row))
print("    K4: median h_2(t) / (17.3 M_{t-1}) by age of the max at t = 1..9 (M_{t-1} > 0.3):",
      [round(float(np.median((H[:, 1:, 2] / (17.3 * Mrun[:, :-1]))[(a1[:, 1:] == a) & (Mrun[:, :-1] > 0.3)])), 3) for a in range(1, 10)])
print("        the same with M_t in the denominator and the least-squares constant instead of 17.3: h_2 ~ c M_t, states with a1 >= 1, M > 0.3: c =",
      round(float((H[:, :, 2] * Mrun)[(a1 >= 1) & (Mrun > 0.3)].sum() / (Mrun ** 2)[(a1 >= 1) & (Mrun > 0.3)].sum()), 2))
del H, Z, theta

# ---------------------------------------------------------------- (c) K6
lam, V = np.linalg.eig(Bc)
k = int(np.argmax(lam.imag))
l = np.linalg.inv(V)[k]
l = l / (l @ (b4 * w[4]))
print("\n(c) K6.  eigenvalue", round(abs(lam[k]), 4), "@", round(float(np.degrees(np.angle(lam[k]))), 2), "deg;  |l . 1| =", round(abs(l @ np.ones(4)), 4),
      ";  |l . b2| =", round(abs(l @ b2), 5), " -> per unit M with h_2 = 17.3 M:", round(abs(l @ b2) * 17.3, 3))
x = sample(300_000, 2424)
H = run(x)
st = running_stats(x)
M, S, a1 = st["M1"], st["M2"], st["a1"]
theta = -(H @ W[COMP].T) / w[COMP]
zc = theta @ l
sel = (a1 >= 1) & (M > 0.3); sel[:, 9] = False
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
print("    age | phase of median | e24: bM, bS, uncentred R2 | centred R2 of the same fit | rms of the perpendicular (imaginary) part / rms of the real part |"
      " complex fit z ~ aM M + aS S+: |aM|@phase, |aS|@phase, uncentred R2, centred R2 | IQR of the phase of z_c (deg), states with (M - S+)/M > 0.3")
for A in range(1, 9):
    s = sel & (a1 == A)
    ph = np.angle(np.median(zc[s].real / M[s]) + 1j * np.median(zc[s].imag / M[s]))
    zr = zc[s] * np.exp(-1j * ph)
    y = zr.real
    X = np.column_stack([M[s], Sp[s]])
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    ru = 1 - ((y - X @ co) ** 2).sum() / (y ** 2).sum()
    rc = 1 - ((y - X @ co) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    perp = np.sqrt(np.mean(zr.imag ** 2) / np.mean(zr.real ** 2))
    cz, *_ = np.linalg.lstsq(X.astype(complex), zc[s], rcond=None)
    resid = zc[s] - X @ cz
    czu = 1 - np.sum(np.abs(resid) ** 2) / np.sum(np.abs(zc[s]) ** 2)
    czc = 1 - np.sum(np.abs(resid) ** 2) / np.sum(np.abs(zc[s] - zc[s].mean()) ** 2)
    big = (M[s] - Sp[s]) / M[s] > 0.3
    dph = np.degrees(np.angle(zr[big]))
    print(f"     {A}  | {np.degrees(ph) % 360:6.1f} | {co[0]:6.3f} {co[1]:7.3f} {ru:.4f} | {rc:.4f} | {perp:.3f} | "
          f"{abs(cz[0]):.3f}@{np.degrees(np.angle(cz[0])) % 360:6.1f}  {abs(cz[1]):.3f}@{np.degrees(np.angle(cz[1])) % 360:6.1f}  {czu:.4f}  {czc:.4f} | "
          f"{np.percentile(dph, 75) - np.percentile(dph, 25):5.1f}   n={s.sum()}")
