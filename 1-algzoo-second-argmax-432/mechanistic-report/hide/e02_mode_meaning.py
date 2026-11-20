"""E02. What do the resting-map mode coordinates hold along real trajectories?

For N(0,1) inputs: mode coordinates m_t of h_t (gauge: record kick = 1), against
M_t, S_t (largest / second largest of x_0..x_t) and the age a1 of the max.
"""
from common import *
from modes import coords, NAMES, LAM, IS_COMPLEX

x = sample(200_000, 11)
H, Z = run(x, return_pre=True)
st = running_stats(x)
m = coords(H)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Mp, Sp = np.maximum(M, 0), np.maximum(np.where(np.isfinite(S), S, 0), 0)


def r2(y, X):
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res.var() / y.var(), beta


print("=== level and runner coordinates (all steps t>=1, no intercept: the net is homogeneous) ===")
sel = np.ones_like(M, bool)
sel[:, 0] = False
for name in ("level", "runner"):
    y = m[name][sel]
    for label, X in (("M+", [Mp[sel]]), ("M+,S+", [Mp[sel], Sp[sel]]), ("M+,S+,x_t", [Mp[sel], Sp[sel], x[sel]])):
        R, b = r2(y, X)
        print(f"  {name:7s} ~ {label:10s}: R2 = {R:.4f}   coef = {np.round(b, 3)}")

print("\n=== level coordinate / (10.157 * M) by age of max (steps with M > 0.3) ===")
for a in range(10):
    s = (a1 == a) & (M > 0.3)
    q = m["level"][s] / (10.157 * M[s])
    print(f"  age {a}: median {np.median(q):.3f}  IQR [{np.percentile(q,25):.3f}, {np.percentile(q,75):.3f}]  n={s.sum()}")

print("\n=== complex modes, normalised by 10.157*M: median amplitude and phase by age of max (M > 0.3) ===")
for name in NAMES:
    if not IS_COMPLEX[name]:
        continue
    print(f"  mode {name}  (free rotation {np.degrees(np.angle(LAM[name])):.2f} deg/step, radius {abs(LAM[name]):.3f})")
    for a in range(10):
        s = (a1 == a) & (M > 0.3)
        q = m[name][s] / (10.157 * M[s])
        med = np.median(q.real) + 1j * np.median(q.imag)
        ph = np.degrees(np.angle(q))
        # circular spread around the median direction
        d = np.degrees(np.angle(q * np.exp(-1j * np.angle(med))))
        print(f"    age {a}: |median| = {abs(med):.3f}  phase = {np.degrees(np.angle(med)):7.1f} deg   "
              f"amp IQR [{np.percentile(abs(q),25):.3f}, {np.percentile(abs(q),75):.3f}]   "
              f"phase IQR [{np.percentile(d,25):6.1f}, {np.percentile(d,75):6.1f}]")

print("\n=== real modes normalised by 10.157*M, by age of max (M > 0.3): median [IQR] ===")
for name in NAMES:
    if IS_COMPLEX[name] or name == "level":
        continue
    row = []
    for a in range(10):
        s = (a1 == a) & (M > 0.3)
        q = m[name][s] / (10.157 * M[s])
        row.append(f"{np.median(q):6.3f}[{np.percentile(q,25):6.3f},{np.percentile(q,75):6.3f}]")
    print(f"  {name:7s}: " + "  ".join(row))
