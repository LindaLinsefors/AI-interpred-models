"""D4 (part 2). Two-pulse inputs against 'centre ~ M, radius ~ M - S', using the impulse's own level l(a) and clock rho(a), phi(a).

For x with the max M at position p1 and the runner-up S at p2 (background b elsewhere), at every step after both pulses:
   level / (10.157 M)                      vs the impulse's l(a1)
   clock z / 10.157                        vs rho(a1) e^{i phi(a1)} (M - S)      -> residual / S  (the S-proportional remainder)
   runner / 10.157 - r(a1) M  divided by S (the S part of the second centre mode)
Also: pre-activations of the pulse units for background 0 (is unit 5 at its threshold?).
python -B d4b_twopulse.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=3, suppress=True, linewidth=220)
oi = np.array(ORD)
f = lambda c: f"{abs(c):5.3f}@{np.degrees(np.angle(c)) % 360:5.1f}"


def modes_of(xv):
    H, Z = run(np.asarray(xv, float), return_pre=True)
    m = coords(H[0])
    return H[0], Z[0], {k: v / 10.157 for k, v in m.items()}


for b in (0.0, -0.5):
    xi = np.full(10, b); xi[0] = 1.0
    Hi, Zi, mi = modes_of(xi)
    print("\n" + "=" * 120)
    print(f"background b = {b}")
    print("  impulse reference x = (1, b, ..., b):  a : level l(a), runner r(a), clock rho(a)@phi(a)")
    print("     " + "  ".join(f"a={a}: {mi['level'][a]:.3f}, {mi['runner'][a]:.3f}, {f(mi['clock'][a])}" for a in range(2, 10)))
    print(f"  pulse pre-activations in the impulse run, t = 1..9:  unit 4: {Zi[1:, 4].round(2)}")
    print(f"                                                       unit 5: {Zi[1:, 5].round(2)}")
    print(f"                                                       unit 9: {Zi[1:, 9].round(2)}")
    for (M, p1, S, p2) in [(2.0, 4, 1.0, 0), (2.0, 0, 1.0, 4), (2.0, 4, 1.8, 0), (2.0, 0, 1.8, 4), (2.0, 4, 0.2, 0), (2.0, 0, 0.2, 4),
                           (2.0, 2, 1.0, 0), (2.0, 0, 1.0, 2), (2.0, 1, 1.0, 0), (2.0, 0, 1.0, 1)]:
        xv = np.full(10, b); xv[p1] = M; xv[p2] = S
        H, Z, m = modes_of(xv)
        t0 = max(p1, p2) + 2
        print(f"  M = {M} at {p1}, S = {S} at {p2}  (x = {xv}); answer of the net: {int(np.argmax(U @ H[9]))} (correct: {p2})")
        print("      t  a1 a2 | level/M (impulse l(a1)) | clock z, prediction rho(a1)(M-S), |z|/pred, residual/S | (runner - r(a1) M)/S | h2/S, alt/S, h3/S | OFF ordinary units")
        for t in range(t0, 10):
            a1, a2 = t - p1, t - p2
            pred = mi["clock"][a1] * (M - S)
            res = (m["clock"][t] - pred) / S
            print(f"      {t}  {a1}  {a2} | {m['level'][t] / M:.3f} ({mi['level'][a1]:.3f}) | {f(m['clock'][t])}  {f(pred)}  {abs(m['clock'][t]) / abs(pred):5.2f}  {f(res)} |"
                  f" {(m['runner'][t] - mi['runner'][a1] * M) / S:6.2f} | {f(m['h2'][t] / S)}  {m['alt'][t] / S:+.2f}  {f(m['h3'][t] / S)} | {[u for u in ORD if Z[t, u] <= 0]}")
