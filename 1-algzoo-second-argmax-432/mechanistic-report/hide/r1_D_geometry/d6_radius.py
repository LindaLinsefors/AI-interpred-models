"""D6. Is the clock radius limited by the ReLUs of the wave units?   python -B d6_radius.py

A. [EXACT] left clock vector per unit: how much one unit of clipped pre-activation in unit i changes each mode coordinate.
B. Impulse (several backgrounds): clock amplitude per age vs free decay, and depth of every clipped ordinary unit.
C. [MEASURED] N(0,1): |z_clock| / (10.157 M) by age and S/M bin, vs 0.951^a.
D. [INTERVENTION] continue from states with a1 = 2 with the wave-unit ReLUs removed (wave units linear) and compare the
   clock amplitude with the real network, by S/M bin.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=3, suppress=True, linewidth=220)
oi = np.array(ORD)
Lc, Ll = LEFT["clock"], LEFT["level"].real
lam = LAM["clock"]

print("A. [EXACT] entries of the left vectors (change of the mode coordinate per unit of activation added to one unit)")
print("   unit    |LEFT_clock|   LEFT_level   LEFT_runner   2|RIGHT_clock| (state radius per unit clock coordinate)   RIGHT_level")
for i, u in enumerate(ORD):
    print(f"   {u:3d}     {abs(Lc[i]):8.4f}     {Ll[i]:8.4f}    {LEFT['runner'].real[i]:8.4f}       {2 * abs(RIGHT['clock'][i]):8.3f}"
          f"                                        {RIGHT['level'].real[i]:8.3f}   {'comparator' if u in COMP else 'wave'}")
ci = [ORD.index(j) for j in COMP]
wi = [ORD.index(j) for j in WAVE]
print(f"   share of |LEFT_clock|^2 on comparators: {(abs(Lc[ci]) ** 2).sum() / (abs(Lc) ** 2).sum():.4f};  max |LEFT_clock| on a wave unit:"
      f" {abs(Lc[wi]).max():.4f} (unit {WAVE[int(np.argmax(abs(Lc[wi])))]}), on a comparator: {abs(Lc[ci]).max():.4f}")
print(f"   share of |LEFT_level|^2 on comparators: {(Ll[ci] ** 2).sum() / (Ll ** 2).sum():.4f}")
# a wave unit clipped at depth d (pre-activation -d) adds d to that unit: relative clock change for depth = its own clock radius
print("   if a wave unit were clipped at a depth equal to its whole clock swing 2|RIGHT_clock_i| |z|, the clock coordinate would change by the fraction"
      " |LEFT_clock_i| * 2|RIGHT_clock_i|:")
print("   " + "  ".join(f"u{u}: {abs(Lc[i]) * 2 * abs(RIGHT['clock'][i]):.3f}" for i, u in enumerate(ORD)))

print("\nB. impulse x = (1, b, ..., b): clock |z|/10.157 per age, ratio to free decay from a = 2, and clipped ordinary units (pre-activation)")
for b in (0.0, -0.25, -0.5, -1.0):
    xv = np.full(10, b); xv[0] = 1.0
    Hh, Zz = run(xv, return_pre=True)
    Hh, Zz = Hh[0], Zz[0]
    z = Hh[:, oi] @ Lc / 10.157
    free = z[2] * lam ** np.arange(-2, 8)
    print(f"   b = {b:5.2f}: |z| a=1..9: {abs(z[1:]).round(3)}")
    print(f"             ratio |z_a| / |z_2 lam^(a-2)|, a=2..9: {(abs(z[2:]) / abs(free[2:])).round(4)};  per-step ratio |z_a/z_(a-1)|: {(abs(z[3:]) / abs(z[2:-1])).round(4)}")
    print(f"             phase advance per step (deg): {np.degrees(np.angle(z[2:] / z[1:-1])).round(2)}")
    clip = [(t, u, round(float(Zz[t, u]), 2)) for t in range(2, 10) for u in ORD if Zz[t, u] <= 0]
    print(f"             clipped ordinary units (age, unit, pre-activation): {clip}")
    lev = Hh[:, oi] @ Ll / 10.157
    print(f"             level/10.157 a=1..9: {lev[1:].round(3)}  per-step ratio a>=3: {(lev[3:] / lev[2:-1]).round(4)}")

print("\nC. [MEASURED] N(0,1), seeds 1031-1034 (4 x 200k): median |z_clock| / (10.157 M) by age a1 and S/M bin (S+ = max(S,0)); M > 0")
BINS = [(-0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
acc = {(b, a): [] for b in range(len(BINS)) for a in range(1, 10)}
accq = {(b, a): [] for b in range(len(BINS)) for a in range(1, 10)}
for seed in range(1031, 1035):
    x = sample(200_000, seed)
    Hh = run(x)
    st = running_stats(x)
    M, S, a1, p1 = st["M1"], st["M2"], st["a1"], st["p1"]
    ratio = np.maximum(np.where(np.isfinite(S), S, 0), 0) / np.where(M > 0, M, 1)
    z = Hh[..., oi] @ Lc / 10.157
    # 'quiet' states: every input after the max is negative
    after_neg = np.ones_like(M, bool)
    for t in range(10):
        later = (np.arange(10)[None, :] > p1[:, t, None]) & (np.arange(10)[None, :] <= t)
        after_neg[:, t] = ~((x > 0) & later).any(1)
    for b, (lo, hi) in enumerate(BINS):
        for a in range(1, 10):
            s = (a1 == a) & (M > 0) & (ratio > lo) & (ratio <= hi)
            acc[b, a].append(abs(z[s]) / M[s])
            accq[b, a].append(abs(z[s & after_neg]) / M[s & after_neg])
print("   S/M bin      " + "".join(f"   a={a}  " for a in range(1, 10)))
for b, (lo, hi) in enumerate(BINS):
    print(f"   ({max(lo, 0):.1f},{hi:.1f}]  " + " ".join(f"{np.median(np.concatenate(acc[b, a])):7.3f}" for a in range(1, 10)))
print("   same, restricted to states where every input after the max is negative ('quiet'); n in brackets for a = 9")
for b, (lo, hi) in enumerate(BINS[:3]):
    print(f"   ({max(lo, 0):.1f},{hi:.1f}]  " + " ".join(f"{np.median(np.concatenate(accq[b, a])) if len(np.concatenate(accq[b, a])) > 30 else np.nan:7.3f}" for a in range(1, 10))
          + f"   [{len(np.concatenate(accq[b, 9]))}]")
print("   free decay 1.02 * 0.9509^(a-2):  " + " ".join(f"{1.02 * 0.9509 ** (a - 2):7.3f}" for a in range(1, 10)))

print("\nD. [INTERVENTION] wave-unit ReLUs removed from age 3 on (states with a1 = 2 at t0; later max unchanged), seed 1035, 400k")


def run_wave_linear(xs, h0):
    h = h0.copy()
    out = np.empty((len(h0), xs.shape[1], 16))
    for t in range(xs.shape[1]):
        zz = xs[:, t, None] * w + h @ W.T
        h = np.maximum(zz, 0)
        h[:, WAVE] = zz[:, WAVE]                      # no ReLU on the wave units
        out[:, t] = h
    return out


x = sample(400_000, 1035)
Hh, Zz = run(x, return_pre=True)
st = running_stats(x)
print("   t0  S/M bin    n      median |z_lin| / |z_real| at k = 1..(9-t0) steps after t0        | median phase(z_lin / z_real) (deg)"
      "       | wave units clipped per step in the real run (mean count)")
for t0 in (2, 4):
    base = (st["a1"][:, t0] == 2) & (st["M1"][:, t0] > 0) & (st["p1"][:, 9] == st["p1"][:, t0])
    Mx = st["M1"][:, 9]
    Sx = np.maximum(st["M2"][:, 9], 0)
    for lo, hi in ((-0.01, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0)):
        s = base & (Sx / Mx > lo) & (Sx / Mx <= hi)
        Hl = run_wave_linear(x[s][:, t0 + 1:], Hh[s, t0])
        zl = Hl[..., oi] @ Lc
        zr = Hh[s][:, t0 + 1:][..., oi] @ Lc
        r = zl / zr
        nclip = (Zz[s][:, t0 + 1:][..., WAVE] <= 0).sum(-1).mean(0)
        print(f"   {t0}  ({max(lo, 0):.1f},{hi:.1f}] {s.sum():6d}   " + " ".join(f"{v:.3f}" for v in np.median(abs(r), 0)) + "   | "
              + " ".join(f"{v:+5.1f}" for v in np.median(np.degrees(np.angle(r)), 0)) + "   | " + " ".join(f"{v:.2f}" for v in nclip))
