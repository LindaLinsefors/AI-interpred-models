"""C2c. Between events the clock rotates freely: z(t) ~ lam z(t-1).  Size of each exact term by age of the max.
400k N(0,1) sequences, seed 1202."""
from cl import *
x, H, Z, st = gen(400_000, 1202)
tm = clock_terms(x, H, Z)
z = tm["z"] / K4
M, S, a1 = st["M1"], pos(st["M2"]), st["a1"]
names = ["carried", "p4", "p5", "p9", "echo", "clip1", "clip6", "clip7", "clip8", "clipwave"]
print("rms of each exact term of z(t) relative to rms |z(t)|, by age a1(t) of the max (states with M > 0, t >= 1)")
print("  a1 |      n  | " + "  ".join(f"{k:>8s}" for k in names) + " | LS z(t) ~ c z(t-1): c, EV | phase advance arg z(t) - arg z(t-1): median [IQR] | same on quiet steps (x_t < S+_(t-1)): c, median advance")
for a in range(1, 10):
    bb, tt = np.nonzero((a1[:, 1:] == a) & (M[:, 1:] > 0)); tt = tt + 1
    zt, zp = z[bb, tt], z[bb, tt - 1]
    r = np.sqrt(np.mean(abs(zt) ** 2))
    c, e = cfit(zt, [zp])
    d = wrap(deg(zt) - deg(zp))
    q = x[bb, tt] < S[bb, tt - 1]
    cq, eq = cfit(zt[q], [zp[q]])
    print(f"  {a}  | {bb.size:7d} | " + "  ".join(f"{np.sqrt(np.mean(abs(tm[k][bb, tt - 1] / K4) ** 2)) / r:8.4f}" for k in names)
          + f" | {fmt(c[0])} {e:.4f} | {np.median(d):+.1f} [{np.percentile(d, 25):+.1f}, {np.percentile(d, 75):+.1f}] | {fmt(cq[0])} (EV {eq:.4f}), {np.median(d[q]):+.1f}")
print(f"\nfree-rotation eigenvalue: {fmt(LAMC, 4)}")
