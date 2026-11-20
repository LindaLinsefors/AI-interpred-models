"""S4: readout columns of the comparators over positions 0..7:
    U[p, j] ~ a_j cos(omega p - phi_j) + c_j,  omega = 42.75 deg (the clock eigen-angle).
Exact (weights only). Also: how specific is the frequency (free-frequency fit, other position ranges),
and the same fit for every other unit's column as a control.
"""
import sys

sys.dont_write_bytecode = True
from vbase import *


def cosfit(y, pos, om):
    G = np.stack([np.cos(om * pos), np.sin(om * pos), np.ones(len(pos))], 1)
    co = np.linalg.lstsq(G, y, rcond=None)[0]
    r = y - G @ co
    return np.hypot(co[0], co[1]), np.degrees(np.arctan2(co[1], co[0])), co[2], 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()


print(f"omega = {np.degrees(OMEGA):.4f} deg")
print("S4  U[p, j] ~ a cos(omega p - phi) + c over p = 0..7     lead: a = 5.81, 5.09, 6.21, 6.08; phi = -29, 26, 131, -130; R^2 = .974, .943, .991, .988")
pos = np.arange(8.0)
for j in COMP:
    a, ph, c, r2 = cosfit(U[:8, j], pos, OMEGA)
    print(f"unit {j}: a = {a:.3f}  phi = {ph:8.2f} deg  c = {c:+.3f}  R^2 = {r2:.4f}")
print("with omega = 42.75 deg exactly:")
for j in COMP:
    a, ph, c, r2 = cosfit(U[:8, j], pos, np.radians(42.75))
    print(f"unit {j}: a = {a:.3f}  phi = {ph:8.2f} deg  c = {c:+.3f}  R^2 = {r2:.4f}")

print("\nfrequency specificity: R^2 of the same 3-parameter fit as a function of the frequency (deg/position), p = 0..7")
oms = np.arange(20, 91, 1.0)
for j in COMP:
    r2s = np.array([cosfit(U[:8, j], pos, np.radians(o))[3] for o in oms])
    fine = np.arange(20, 90, 0.05)
    r2f = np.array([cosfit(U[:8, j], pos, np.radians(o))[3] for o in fine])
    print(f"unit {j}: best frequency {fine[r2f.argmax()]:.2f} deg (R^2 {r2f.max():.4f});  R^2 at 30/35/40/42.75/45/50/55/60: "
          + " ".join(f"{cosfit(U[:8, j], pos, np.radians(o))[3]:.3f}" for o in (30, 35, 40, 42.75, 45, 50, 55, 60)))
print("joint best frequency for the four columns (sum of SS_res):")
fine = np.arange(20, 90, 0.05)
tot = []
for o in fine:
    G = np.stack([np.cos(np.radians(o) * pos), np.sin(np.radians(o) * pos), np.ones(8)], 1)
    Y = U[:8][:, COMP]
    r = Y - G @ np.linalg.lstsq(G, Y, rcond=None)[0]
    tot.append((r ** 2).sum())
tot = np.array(tot)
Yc = U[:8][:, COMP] - U[:8][:, COMP].mean(0)
print(f"  best {fine[tot.argmin()]:.2f} deg, pooled R^2 {1 - tot.min() / (Yc ** 2).sum():.4f}; at 42.75: "
      f"{1 - tot[np.abs(fine - 42.75).argmin()] / (Yc ** 2).sum():.4f}")

print("\nnull for the fit: a 3-parameter sinusoid fit to 8 iid N(0,1) points at 42.75 deg has R^2 (median, 95th, 99th pct):")
rng = np.random.default_rng(5999)
Y = rng.standard_normal((200_000, 8))
G = np.stack([np.cos(OMEGA * pos), np.sin(OMEGA * pos), np.ones(8)], 1)
r = Y - (Y @ np.linalg.pinv(G).T) @ G.T
r2 = 1 - (r ** 2).sum(1) / ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
print("  ", np.percentile(r2, [50, 95, 99]))

print("\nsame fit over positions 0..8 and 0..9 (the claim uses 0..7):")
for hi in (9, 10):
    p = np.arange(float(hi))
    print(f"  p = 0..{hi - 1}: " + "  ".join(f"unit {j}: a {cosfit(U[:hi, j], p, OMEGA)[0]:.2f} R^2 {cosfit(U[:hi, j], p, OMEGA)[3]:.3f}" for j in COMP))

print("\ncontrol: the same fit (p = 0..7, 42.75 deg) for every unit's readout column")
for j in range(N):
    a, ph, c, r2 = cosfit(U[:8, j], pos, OMEGA)
    grp = "COMP" if j in COMP else "PULSE" if j in PULSE else "wave"
    print(f"unit {j:2d} ({grp:5s}): a = {a:5.2f}  phi = {ph:7.1f}  c = {c:+6.2f}  R^2 = {r2:.3f}")
print("\nU[:, COMP] (rows = positions 0..9):")
print(U[:, COMP])

# How much of V over i = 0..7 is a 42.75-degree sinusoid + offset for ARBITRARY non-negative comparator states?
rng = np.random.default_rng(5998)
hc = rng.exponential(1.0, (200_000, 4))
Vr = hc @ UC[:8].T
r = Vr - (Vr @ np.linalg.pinv(G).T) @ G.T
frac = 1 - (r ** 2).sum(1) / ((Vr - Vr.mean(1, keepdims=True)) ** 2).sum(1)
print("\nV[0..7] for random comparator states h_c ~ iid Exp(1) (nothing to do with the network's dynamics):")
print("  fraction of the variance over positions that is an omega-sinusoid: median %.4f, 5th pct %.4f, 1st pct %.4f"
      % tuple(np.percentile(frac, [50, 5, 1])))
print("  pooled: %.4f" % (1 - (r ** 2).sum() / ((Vr - Vr.mean(1, keepdims=True)) ** 2).sum()))
