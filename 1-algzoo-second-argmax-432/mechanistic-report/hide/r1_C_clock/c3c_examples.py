"""C3c. Two deterministic examples: a runner-up v = 0.6 arriving 1 step or 6 steps after the max M = 1.5 (background -1)."""
from cl import *
for p2 in (1, 2, 3, 4, 5, 6, 7, 8):
    xb = np.full(10, -1.0); xb[0] = 1.5
    xe = xb.copy(); xe[p2] = 0.6
    Hb = run(xb)[0]; He, Ze = run(xe, return_pre=True); He, Ze = He[0], Ze[0]
    th = thresholds(Hb[p2 - 1])
    zb, ze = Hb[:, ORD] @ LC / K4, He[:, ORD] @ LC / K4
    clipped = [j for j in COMP if Ze[p2, j] <= 0]
    lg = U @ He[9]
    late = min(p2 + 2, 9)
    print(f"v at position {p2} (max-age {p2}): thresholds theta_(1,6,7,8) = {np.round(th[:4], 3)}, theta_4 = {th[4]:.3f} | clipped: {clipped}, unit 4 {'fires' if Ze[p2, 4] > 0 else 'silent'}"
          f" | z/K4 without v: {fmt(zb[p2])}, with v: {fmt(ze[p2])}; F6 target (M - v) = 0.9 vs M = 1.5: |z| ratio {abs(ze[p2]) / abs(zb[p2]):.3f} (target 0.600), two steps later {abs(ze[late]) / abs(zb[late]):.3f}"
          f" | units 5, 9 at the next step: {He[min(p2 + 1, 9), 5]:.2f}, {He[min(p2 + 1, 9), 9]:.2f} | answer {lg.argmax()} (correct {p2})")
