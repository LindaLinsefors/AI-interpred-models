"""D1 (check). 'Directions on the circle have a fixed meaning': decode the age of the max from the clock phase alone.

z = LEFT_clock . h_t[ORD].  Reference phases phi(a) = phase of the impulse response (x = (1, -0.5, ..., -0.5)) at age a.
Decoder: a_hat = argmin_a |angle(z e^{-i phi(a)})| over a = 1..9.   N(0,1), seeds 1071..1074 (4 x 200k), states with M > 0, a1 >= 1.
python -B d1_phase.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

oi = np.array(ORD)
Lc = LEFT["clock"]
xi = np.full(10, -0.5); xi[0] = 1.0
zi = run(xi)[0][:, oi] @ Lc
PHI = np.angle(zi)                                     # phi(a), a = 0..9 (a = 0 unused)
print("reference phases phi(a), a = 1..9 (deg):", np.round(np.degrees(PHI[1:]) % 360, 1))
BINS = [(-0.01, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0)]
err = {b: {a: [] for a in range(1, 10)} for b in range(len(BINS))}
hit = {b: {a: [] for a in range(1, 10)} for b in range(len(BINS))}
for seed in range(1071, 1075):
    x = sample(200_000, seed)
    H = run(x)
    st = running_stats(x)
    M, S, a1 = st["M1"], st["M2"], st["a1"]
    r = np.maximum(np.where(np.isfinite(S), S, 0), 0) / np.where(M > 0, M, 1)
    z = H[..., oi] @ Lc
    dist = np.abs(np.angle(z[..., None] * np.exp(-1j * PHI[1:])))           # (B, T, 9)
    ahat = dist.argmin(-1) + 1
    for b, (lo, hi) in enumerate(BINS):
        for a in range(1, 10):
            s = (a1 == a) & (M > 0) & (r > lo) & (r <= hi)
            err[b][a].append(np.degrees(np.angle(z[s] * np.exp(-1j * PHI[a]))))
            hit[b][a].append(ahat[s] == a)
print("\nmedian |phase error| (deg) of the clock coordinate relative to phi(a1); rows S+/M bin, columns a1 = 1..9, last column: all a1 >= 2")
for b, (lo, hi) in enumerate(BINS):
    e = {a: np.concatenate(err[b][a]) for a in range(1, 10)}
    print(f"   ({max(lo, 0):.1f},{hi:.1f}] | " + " ".join(f"{np.median(abs(e[a])):6.1f}" for a in range(1, 10))
          + f" | {np.median(abs(np.concatenate([e[a] for a in range(2, 10)]))):6.1f}")
print("\nmean signed phase error (deg)")
for b, (lo, hi) in enumerate(BINS):
    e = {a: np.concatenate(err[b][a]) for a in range(1, 10)}
    print(f"   ({max(lo, 0):.1f},{hi:.1f}] | " + " ".join(f"{np.degrees(np.angle(np.exp(1j * np.radians(e[a])).mean())):+6.1f}" for a in range(1, 10)))
print("\naccuracy of the nearest-phase decoder of a1 (9 classes, chance 0.11); last column: all a1 >= 1")
for b, (lo, hi) in enumerate(BINS):
    h = {a: np.concatenate(hit[b][a]) for a in range(1, 10)}
    print(f"   ({max(lo, 0):.1f},{hi:.1f}] | " + " ".join(f"{h[a].mean():6.3f}" for a in range(1, 10)) + f" | {np.concatenate(list(h.values())).mean():6.3f}"
          + f"   n = {sum(len(v) for v in h.values())}")
