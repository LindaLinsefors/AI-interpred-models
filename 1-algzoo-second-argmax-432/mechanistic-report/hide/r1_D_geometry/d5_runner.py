"""D5. Does the runner-up have a circle too?   python -B d5_runner.py

A. From the per-(a1, a2) regressions h_t[ORD] ~ g u + S v + T tau + x e (d1_cells_strict.npz): for each oscillating mode m,
   the coordinate c_m(a1, a2) of the S coefficient, rotated back by the free rotation since the more recent of the two events,
       F_m(a1, a2) = c_m(a1, a2) * lam_m^-(min(a1, a2) - 2),
   listed by d = a1 - a2 (d > 0: runner-up arrived d steps after the max; d < 0: the runner-up is the old max).
   If the component rotates freely since the last top-2 event, F depends on d only.
B. Ratios tau/v and u/v of the mode coordinates: is the amplitude proportional to S, M - S or S - T?
C. Check of the regression against exact traces: mean of (J_p1 + J_p2) and J_p3 over the states of a cell (seed 1051, 60k sequences).
D. How big is this component in state space, compared with the max's clock?
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "d1_cells_strict.npz", allow_pickle=True)
B, n, regs = d["B"], d["n"], list(d["regs"])
kg, kS, kT = regs.index("g"), regs.index("S"), regs.index("T")
OSC = ["h2", "h3", "alt", "short", "clock"]
fmt = lambda c: f"{abs(c):5.2f}@{np.degrees(np.angle(c)) % 360:4.0f}"

print("A. F_m(a1, a2) = [mode coordinate of the S coefficient] / 10.157, rotated back to min(a1, a2) = 2.   Cells with a1, a2 >= 2.")
summary = {}
for m in OSC:
    lam = LAM[m]
    print(f"\n   mode {m}: |lam| = {abs(lam):.3f}, angle = {np.degrees(np.angle(lam)):.2f} deg")
    print("     d = a1-a2 | F for each cell with min(a1,a2) = 2, 3, 4, ...                                         | mean F  | spread |F - mean| / |mean|")
    for dd in list(range(-7, 0)) + list(range(1, 8)):
        vals, wts = [], []
        for lo in range(2, 10):
            a1, a2 = (lo + dd, lo) if dd > 0 else (lo, lo - dd)
            if a1 > 9 or a2 > 9 or np.isnan(B[a1, a2, kS, 0]):
                continue
            c = (B[a1, a2, kS] @ LEFT[m]) / 10.157
            vals.append(c * lam ** -(lo - 2))
            wts.append(n[a1, a2])
        vals, wts = np.array(vals), np.array(wts, float)
        mean = (vals * wts).sum() / wts.sum()
        spread = np.sqrt((abs(vals - mean) ** 2 * wts).sum() / wts.sum()) / abs(mean)
        summary[m, dd] = mean
        print(f"     {dd:+d}        | " + "  ".join(fmt(v) for v in vals).ljust(95) + f"| {fmt(mean)} | {spread:.2f}")
    print("     |F(d)| by |d| (d>0 / d<0): " + "   ".join(f"|d|={k}: {abs(summary[m, k]):.2f} / {abs(summary[m, -k]):.2f}" for k in range(1, 8)))
    print("     phase difference F(d) vs F(-d) (deg): " + "  ".join(f"|d|={k}: {np.degrees(np.angle(summary[m, k] / summary[m, -k])):+.0f}" for k in range(1, 8)))

print("\nB. Ratios of mode coordinates of the coefficients, cells with a1, a2 >= 2 and |a1 - a2| >= 4 (weighted by cell size):")
print("   tau/v = -1 would mean 'proportional to S - T';  u/v = 0 means no dependence on M at fixed S and T.")
for m in OSC:
    num_t = num_u = den = 0
    rt, ru = [], []
    for a1 in range(2, 10):
        for a2 in range(2, 10):
            if abs(a1 - a2) < 4 or np.isnan(B[a1, a2, kS, 0]):
                continue
            v, t, u = (B[a1, a2, k] @ LEFT[m] for k in (kS, kT, kg))
            num_t += n[a1, a2] * t * np.conj(v); num_u += n[a1, a2] * u * np.conj(v); den += n[a1, a2] * abs(v) ** 2
            rt.append(t / v); ru.append(u / v)
    rt, ru = np.array(rt), np.array(ru)
    print(f"   {m:6s}: tau/v = {fmt(num_t / den)} (range of |.| over cells {abs(rt).min():.2f}-{abs(rt).max():.2f});   u/v = {fmt(num_u / den)}"
          f" (range of |.| {abs(ru).min():.2f}-{abs(ru).max():.2f})")

print("\nC. Exact traces vs regression (seed 1051, 60k sequences, states with S > 0): mode coordinate / 10.157 of")
print("   mean(J_p1 + J_p2) [regression: v],  mean(J_p1) [u],  mean(J_p3) [tau]   for selected cells")
x = sample(60_000, 1051)
H, J = traces(x)
st = running_stats(x)
bi = np.arange(len(x))
for (a1, a2) in [(8, 2), (9, 3), (7, 4), (2, 8), (3, 9), (4, 9), (5, 4), (6, 4), (4, 5), (9, 7)]:
    rows = []
    for t in range(max(a1, a2), 10):
        s = (st["a1"][:, t] == a1) & (st["a2"][:, t] == a2) & (st["M2"][:, t] > 0) & (st["p3"][:, t] >= 0)
        if s.sum():
            idx = bi[s]
            rows.append((J[idx, t, st["p1"][idx, t]][:, ORD], J[idx, t, st["p2"][idx, t]][:, ORD], J[idx, t, st["p3"][idx, t]][:, ORD]))
    J1, J2, J3 = (np.concatenate([r[k] for r in rows]) for k in range(3))
    line = f"   cell (a1={a1}, a2={a2}), n = {len(J1):5d}: "
    for m in ("h2", "clock"):
        L = LEFT[m] / 10.157
        line += (f"{m}: v trace {fmt((J1 + J2).mean(0) @ L)} reg {fmt(B[a1, a2, kS] @ L)} | u trace {fmt(J1.mean(0) @ L)} reg {fmt(B[a1, a2, kg] @ L)}"
                 f" | tau trace {fmt(J3.mean(0) @ L)} reg {fmt(B[a1, a2, kT] @ L)}   ")
    print(line)

print("\nD. Size in state space: rms (over phase) 13-dim norm of each mode's part of the state per unit of S, 10.157 * |F| * sqrt(2) * |RIGHT_m|")
print("   (for the real mode alt: 10.157 * |F| * |RIGHT|), at min(a1, a2) = 2, compared with the max's clock per unit of M - S (|u_clock| ~ 1.02 at a1 = 2)")
for m in OSC:
    k = np.sqrt(2) * np.linalg.norm(RIGHT[m]) if IS_COMPLEX[m] else np.linalg.norm(RIGHT[m])
    print(f"   {m:6s}: norm factor {k:.2f};  |d| = 1: {10.157 * abs(summary[m, 1]) * k:5.1f}   |d| = 2: {10.157 * abs(summary[m, 2]) * k:5.1f}   |d| = 4: {10.157 * abs(summary[m, 4]) * k:5.1f}"
          f"   |d| = 6: {10.157 * abs(summary[m, 6]) * k:5.1f}    decay over 7 steps: {abs(LAM[m]) ** 7:.2f}")
print(f"   max's clock per unit (M - S) at a1 = 2: {10.157 * 1.02 * np.sqrt(2) * np.linalg.norm(RIGHT['clock']):.1f}")
