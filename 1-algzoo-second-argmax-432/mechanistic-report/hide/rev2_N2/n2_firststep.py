"""N2: the clock's first step after a new maximum (section 5.4 'exactly' formula vs section 4.1's 36 deg)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import LEFT, LAM, coords

w4 = w[4]; ang = lambda z: np.degrees(np.angle(z))
L = LEFT["clock"]; i0 = ORD.index(0)
f = LAM["clock"] + (L @ W[ORD, 5]) * W[5, 4] + (L @ W[ORD, 9]) * W[9, 4]
print("pulses only (report's formula): %.3f@%.1f ; LEFT_clock[unit 0] = %.3f@%.1f" % (abs(f), ang(f), abs(L[i0]), ang(L[i0])))
# impulse
for xs, lab in ((np.eye(10)[0], "x=(1,0,...,0)"), (np.r_[1.0, -1 * np.ones(9)], "x=(1,-1,...,-1)")):
    H, Z = run(xs, return_pre=True); z = coords(H[0])["clock"] / w4
    clip0 = -min(Z[0, 2, 0], 0)
    print(f"{lab}: z(1) {abs(z[1]):.3f}@{ang(z[1]):.1f}  z(2) {abs(z[2]):.3f}@{ang(z[2]):.1f}  ratio {abs(z[2]/z[1]):.3f}@{ang(z[2]/z[1]):.1f} ; unit 0 pre-activation at t=2: {Z[0,2,0]:.2f} ; formula + unit-0 clip term: ", end="")
    g = f + L[i0] * clip0 / H[0, 0, 4]
    print("%.3f@%.1f" % (abs(g), ang(g)))
# data: new max at s (over positive old max, old max >= 2 steps old), next two inputs below the old runner-up
x = sample(400_000, 9221); H, Z = run(x, return_pre=True); st = running_stats(x); z = coords(H)["clock"] / w4
r = []
for s in range(2, 7):
    M0, S0 = st["M1"][:, s - 1], st["M2"][:, s - 1]
    ok = (x[:, s] > M0) & (M0 > 0.3) & (x[:, s + 1] < np.maximum(S0, 0)) & (x[:, s + 2] < np.maximum(S0, 0)) & ((x[:, s] - M0) > 0.5 * M0)
    r.append(z[ok, s + 2] / z[ok, s + 1])
r = np.concatenate(r)
print("data, new max exceeding the old by > 50%%, two low followers (n=%d): median |z(s+2)/z(s+1)| = %.3f, median angle %.1f" % (len(r), np.median(abs(r)), np.median(ang(r))))
