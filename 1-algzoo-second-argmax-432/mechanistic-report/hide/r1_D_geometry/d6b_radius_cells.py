"""D6b / D1. Clock radius by (a1, a2) cell and S/M: is it proportional to M - S in every cell?
Median of |z_clock| / (10.157 M) and circular mean of the phase offset from the cell-independent phase phi(a1), N(0,1), seeds 1061..1064.
python -B d6b_radius_cells.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

oi = np.array(ORD)
Lc = LEFT["clock"]
PHI = np.radians([0, 0, 36, 79, 123, 166, 209, 251, 294, 337])          # phase of the clock at age a (measured, small S/M)
BINS = [(-0.01, 0.1), (0.4, 0.6), (0.9, 1.0)]
val = {(b, a1, a2): [] for b in range(3) for a1 in range(2, 10) for a2 in range(10)}
for seed in range(1061, 1065):
    x = sample(200_000, seed)
    H = run(x)
    st = running_stats(x)
    M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
    r = np.maximum(np.where(np.isfinite(S), S, 0), 0) / np.where(M > 0, M, 1)
    z = H[..., oi] @ Lc / 10.157
    for b, (lo, hi) in enumerate(BINS):
        for A1 in range(2, 10):
            base = (a1 == A1) & (M > 0) & (r > lo) & (r <= hi)
            for A2 in range(10):
                s = base & (a2 == A2)
                if s.any():
                    val[b, A1, A2].append(z[s] / M[s] * np.exp(-1j * PHI[A1]))
for b, (lo, hi) in enumerate(BINS):
    print(f"\nS+/M in ({max(lo, 0):.1f}, {hi:.1f}]: median |z| / (10.157 M) and mean phase offset from phi(a1) (deg); rows a1 = 2..9, columns a2 = 0..9; '.' = fewer than 200 states")
    for A1 in range(2, 10):
        row = ""
        for A2 in range(10):
            v = np.concatenate(val[b, A1, A2]) if val[b, A1, A2] else np.array([])
            row += f"  {np.median(abs(v)):.2f}{np.degrees(np.angle(v.mean())):+5.0f}" if len(v) >= 200 else "       .   "
        print(f"   a1={A1} |{row}")
