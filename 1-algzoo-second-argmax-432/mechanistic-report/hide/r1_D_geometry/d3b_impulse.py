"""D3 (deterministic illustration + exact checks).   python -B d3b_impulse.py

1. [EXACT] ingredients of the linear-algebra argument.
2. Impulse x = (1, b, ..., b): scale the level coordinate at t0 = 3 by f and continue; clock amplitude / phase / level and clipped units.
3. How often is the gate pattern exactly 'resting' (13 ordinary units on, 3 pulses off) on N(0,1) data?  (seed 1081, 200k)
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=4, suppress=True, linewidth=200)
oi = np.array(ORD)
Lc, Ll, Rl = LEFT["clock"], LEFT["level"].real, RIGHT["level"].real

print("1. [EXACT]")
ev = np.linalg.eigvals(A)
print("   eigenvalues of A = W[ORD, ORD] sorted by |.|:", np.round(sorted(ev, key=lambda v: -abs(v)), 4))
real = sorted([v.real for v in ev if abs(v.imag) < 1e-9], reverse=True)
print("   real eigenvalues:", np.round(real, 4), " -> distance of the largest from 1:", round(real[0] - 1, 5))
print("   level eigenvector (unit norm):", np.round(Rl / np.linalg.norm(Rl), 3), " all > 0:", bool((Rl > 0).all()))
print("   runner eigenvector (gauge):   ", np.round(RIGHT["runner"].real, 3))
print("   A has negative entries:", int((A < 0).sum()), "of 169;  so positivity of the level eigenvector is not Perron-Frobenius for a non-negative matrix")
print("   |A c - c| / |c| for c = level eigenvector:", np.linalg.norm(A @ Rl - Rl) / np.linalg.norm(Rl), "(= lambda - 1)")
print("   2D rotation block of the clock: |lam| = %.4f, angle = %.3f deg;  (I - R) c = 0 has only c = 0 inside the plane: det(I - R_2x2) = %.4f"
      % (abs(LAM["clock"]), np.degrees(np.angle(LAM["clock"])), abs(1 - LAM["clock"]) ** 2))
print("   coordinates of the input direction w[ORD] in level, runner, clock:", np.round([Ll @ w[oi], LEFT["runner"].real @ w[oi]], 4), np.round(Lc @ w[oi], 4))
print("   level growth over 7 resting steps: %.4f;  clock decay over 7 steps: %.4f;  radius/centre shrinks by %.4f per step"
      % (LAM["level"].real ** 7, abs(LAM["clock"]) ** 7, abs(LAM["clock"]) / LAM["level"].real))

print("\n2. impulse, level coordinate scaled by f at t0 = 3 (pure coordinate scaling), then free run with the same inputs")
for b in (0.0, -0.5):
    xv = np.full(10, b); xv[0] = 1.0
    H0, Z0 = run(xv, return_pre=True)
    H0, Z0 = H0[0], Z0[0]
    z0, l0 = H0[:, oi] @ Lc, H0[:, oi] @ Ll
    print(f"   b = {b}")
    for f in (0.0, 0.5, 0.75, 1.25):
        h = H0[3].copy()
        h[oi] += (f - 1) * l0[3] * Rl
        Hp, Zp = run(xv[4:], return_pre=True, h0=h)
        Hp, Zp = Hp[0], Zp[0]
        zp, lp = Hp[:, oi] @ Lc, Hp[:, oi] @ Ll
        r = zp / z0[4:]
        print(f"     f = {f:4.2f}: negative entries at t0: {[u for u in ORD if h[u] < 0]}")
        print(f"        |z'/z| t=4..9: {np.round(abs(r), 3)}   phase diff (deg): {np.round(np.degrees(np.angle(r)), 1)}   level'/level: {np.round(lp / l0[4:], 3)}")
        print(f"        OFF ordinary units per step (baseline -> patched): " + "; ".join(
            f"t={t}: {[u for u in ORD if Z0[t, u] <= 0]} -> {[u for u in ORD if Zp[t - 4, u] <= 0]}" for t in range(4, 10)))
        print(f"        pulses firing in the patched run (t, unit, value): {[(t + 4, u, round(float(Hp[t, u]), 2)) for t in range(6) for u in PULSE if Hp[t, u] > 0]}")

print("\n3. [MEASURED] N(0,1), seed 1081, 200k sequences: share of states whose gate pattern is exactly resting (13 ordinary on, pulses off)")
x = sample(200_000, 1081)
H, Z = run(x, return_pre=True)
st = running_stats(x)
G = Z > 0
rest = G[..., oi].all(-1) & ~G[..., PULSE].any(-1)
ord_on = G[..., oi].all(-1)
for a in range(10):
    s = st["a1"] == a
    print(f"   a1 = {a}: resting {rest[s].mean():.3f};  all 13 ordinary on {ord_on[s].mean():.3f};  mean number of ordinary units off {(~G[..., oi])[s].sum(-1).mean():.2f};"
          f"  unit 5 on {G[..., 5][s].mean():.3f};  unit 4 on {G[..., 4][s].mean():.3f};  unit 9 on {G[..., 9][s].mean():.3f}")
s = st["a1"] >= 2
print(f"   all a1 >= 2: resting {rest[s].mean():.3f};  all 13 ordinary on {ord_on[s].mean():.3f}")
# how well does one resting step predict the next ordinary state?  h_{t+1}[ORD] vs A h_t[ORD] + W[ORD,PULSE] h_t[PULSE] + w x_{t+1}
pred = H[:, :-1][..., oi] @ A.T + H[:, :-1][..., PULSE] @ W[np.ix_(ORD, PULSE)].T + x[:, 1:, None] * w[oi]
act = H[:, 1:][..., oi]
s2 = st["a1"][:, 1:] >= 3
print(f"   states with a1 >= 3: exact linear step (pre-activation) equals the state for {np.isclose(pred, act, atol=1e-9).all(-1)[s2].mean():.3f} of them;"
      f"  rms |ReLU correction| / rms |state| = {np.sqrt(((act - pred)[s2] ** 2).sum(-1).mean()) / np.sqrt((act[s2] ** 2).sum(-1).mean()):.3f}")
