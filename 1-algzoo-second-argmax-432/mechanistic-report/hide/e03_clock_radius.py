"""E03. Does the clock radius encode the gap M - S?  And where is the centre?

z = clock coordinate (gauge: record kick = 1).  For each age a of the max we rotate z back by the
median phase at that age and regress its real part (the 'radius along the expected direction')
on M and S (no intercept; the net is homogeneous).  S+ = max(S, 0) since negative values are ~not stored.
"""
from common import *
from modes import coords, NAMES, LAM, IS_COMPLEX

x = sample(400_000, 12)
H = run(x)
st = running_stats(x)
m = coords(H)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
z = m["clock"] / 10.157

def fit(y, cols):
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()   # uncentred R2 (no intercept)

print("clock: Re(z e^{-i phi_a}) ~ b_M M + b_S S+   (steps with M > 0), and the same for the level coordinate")
print(" age   phi_a    b_M     b_S    R2(M,S+)   R2(M only)   R2(M-S+ only) |   level: b_M   b_S   R2")
for a in range(1, 10):
    s = (a1 == a) & (M > 0)
    phi = np.angle(np.median(z[s].real / M[s]) + 1j * np.median(z[s].imag / M[s]))
    y = (z[s] * np.exp(-1j * phi)).real
    b, R = fit(y, [M[s], Sp[s]])
    b1, R1 = fit(y, [M[s]])
    b2, R2 = fit(y, [M[s] - Sp[s]])
    bl, Rl = fit(m["level"][s] / 10.157, [M[s], Sp[s]])
    print(f"  {a}   {np.degrees(phi):7.1f}  {b[0]:6.3f}  {b[1]:6.3f}   {R:.4f}     {R1:.4f}       {R2:.4f} (b={b2[0]:.3f}) |"
          f"   {bl[0]:6.3f} {bl[1]:6.3f}  {Rl:.4f}")

print("\nsplit by whether the runner-up arrived before the max (a2 > a1: it is the old max) or after (a2 < a1)")
for label, cond in (("runner-up older than max (old max)", a2 > a1), ("runner-up younger than max", (a2 < a1) & (a2 >= 0))):
    print(" ", label)
    for a in range(1, 10):
        s = (a1 == a) & (M > 0) & cond & (S > 0)
        if s.sum() < 500:
            continue
        phi = np.angle(np.median(z[s].real / M[s]) + 1j * np.median(z[s].imag / M[s]))
        y = (z[s] * np.exp(-1j * phi)).real
        b, R = fit(y, [M[s], Sp[s]])
        yi = (z[s] * np.exp(-1j * phi)).imag
        bi, Ri = fit(yi, [M[s], Sp[s]])
        print(f"    age {a}: phi = {np.degrees(phi):7.1f}  radial: b_M = {b[0]:6.3f}  b_S = {b[1]:6.3f}  R2 = {R:.4f}"
              f"   | tangential: b_M = {bi[0]:6.3f} b_S = {bi[1]:6.3f}   n = {s.sum()}")
