"""Reviewer R: three small weight-level checks behind suggested rewrites. Reads weights only, writes nothing."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *          # noqa: E402  (w, W, U, COMP, WAVE, ORD, run, np)
import modes                  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=160)
g = -w[COMP]                                            # comparator gains g_j = -w_j
w4 = w[4]
sat = [u for u in range(16) if u not in COMP + [2, 4]]  # the ten satellites

# (a) what the thresholds one step after a new maximum are made of (example of section 5.5)
x = np.array([0.8, -0.5, 0.3, 1.2, -1.0, 0.9, -0.7, 1.6, -0.2, 0.4])
H = run(x)[0]
print("(a) theta(t+1) = Bc[theta-x]+ + b2 h2 + b4 h4 + eps, example of 5.5")
for t in [3, 7]:                                        # steps of the 2nd and 3rd new maximum
    h = H[t]
    theta = (W[COMP] @ h) / g
    parts = {
        "comparator block": (W[np.ix_(COMP, COMP)] @ h[COMP]) / g,
        "unit 2 (b2 h2)": W[COMP, 2] * h[2] / g,
        "unit 4 (b4 h4)": W[COMP, 4] * h[4] / g,
        "satellites (eps)": (W[np.ix_(COMP, sat)] @ h[sat]) / g,
    }
    M0 = x[:t].max()
    print(f"  step {t}: x_t = {x[t]}, old max M0 = {M0}, h2 = {h[2]:.3f} = {h[2] / M0:.2f} M0, h4 = {h[4]:.3f}")
    print(f"    theta(t+1) for units 1,6,7,8: {theta}")
    for k, v in parts.items():
        print(f"    {k:18s} {v}   (/M0: {v / M0})")
    th4 = -(W[4] @ h) / w4
    print(f"    theta_4(t+1) = {th4:.3f}; from unit 2 {-W[4, 2] * h[2] / w4:.3f}, unit 4 {-W[4, 4] * h[4] / w4:.3f}, "
          f"comparators {-(W[4, COMP] @ h[COMP]) / w4:.3f}, satellites {-(W[4, sat] @ h[sat]) / w4:.3f}")

# (b) sizes of the level and runner vectors in the report's gauge
rl, rr = modes.RIGHT["level"].real, modes.RIGHT["runner"].real
print("\n(b) |r_level| =", np.linalg.norm(rl), " |r_runner| =", np.linalg.norm(rr), " ratio runner/level =", np.linalg.norm(rr) / np.linalg.norm(rl))
print("    cos(r_level, r_runner) =", rl @ rr / np.linalg.norm(rl) / np.linalg.norm(rr))
print("    units", ORD)
print("    r_level ", rl)
print("    r_runner", rr)

# (c) how one step writes into the runner (and level, clock) coordinate, per unit of each source, in units of w4
print("\n(c) writes into mode coordinates, divided by w4")
for name in ["level", "runner", "clock"]:
    L = modes.LEFT[name]
    Lfull = np.zeros(16, dtype=complex)
    Lfull[ORD] = L
    direct = L @ w[ORD]                                  # one unit of input, all 13 units on
    cut = Lfull[COMP] * g                                # a cut comparator k adds g_k (x - theta_k) to unit k
    p5, p9 = L @ W[ORD, 5], L @ W[ORD, 9]                # one unit of activation of unit 5 / unit 9, one step later
    fmt = (lambda v: f"{abs(v):.3f}@{np.degrees(np.angle(v)):.0f}deg") if modes.IS_COMPLEX[name] else (lambda v: f"{v.real:+.3f}")
    print(f"  {name:6s} direct input {fmt(direct / w4)} | cut comparator 1,6,7,8: {[fmt(c / w4) for c in cut]}"
          f" | unit 5 per unit activation {fmt(p5)} (x18.009/w4 = {fmt(p5 * 18.009 / w4)}) | unit 9 {fmt(p9)}")
