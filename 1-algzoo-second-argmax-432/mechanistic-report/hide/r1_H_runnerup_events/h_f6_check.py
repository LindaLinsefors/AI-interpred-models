"""Re-check of F5 / F6 (level and clock laws) on a fresh seed."""
from hcommon import *

x = sample(300_000, 1041)
H = run(x); st = running_stats(x); m = coords(H)
M, S, a1 = st["M1"], st["M2"], st["a1"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
print("re-check of F5/F6 (seed 1041): level/(K M) and clock/(K (M - S+)) by age of the max at the same step, M > 0.3, gap > 0.2 M")
print("  a | level/(K M) median | clock: |median| @ phase of z/(K (M-S+)) | phase IQR")
for a in range(0, 10):
    s = (a1 == a) & (M > 0.3) & ((M - Sp) > 0.2 * M)
    q = m["clock"][s] / (K * (M[s] - Sp[s]))
    med = np.median(q.real) + 1j * np.median(q.imag)
    dph = np.degrees(np.angle(q * np.exp(-1j * np.angle(med))))
    print(f"  {a} | {np.median(m['level'][s] / (K * M[s])):.3f} | {abs(med):.3f} @ {np.degrees(np.angle(med)) % 360:6.1f} | [{np.percentile(dph, 25):5.1f}, {np.percentile(dph, 75):5.1f}]   n={s.sum()}")
