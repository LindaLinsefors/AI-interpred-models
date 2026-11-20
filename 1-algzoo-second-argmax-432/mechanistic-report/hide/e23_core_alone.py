"""E23. How autonomous is the core?  Simulate only units {2, 4, 1, 6, 7, 8}
    h_core' = ReLU(W[core, core] h_core + w[core] x)
with the other ten units removed (variant A), or with the other units' feedback replaced by a fixed
multiple of unit 2 (variant B: the wave units' level is slaved to unit 2 along the level eigenvector).
Compare with the real network: centred R2 of the unit activations over all steps, gate agreement and the
firing of unit 4.  (Thresholds are not compared here.)
Yardstick for those R2, a model with no dynamics at all: every threshold is a constant times the running max
of the earlier inputs, theta_j(t) = c_j Mp(t) with Mp(t) = max(0, x_0..x_(t-1)), so h_j = ReLU(w_j (x_t - c_j Mp));
for unit 2 that is h_2 = ReLU(w_2 x_t + 17.3 Mp)."""
from common import *
from modes import RIGHT

core = [2, 4, 1, 6, 7, 8]
rest = [u for u in range(16) if u not in core]
x = sample(300_000, 2323)
H, Z = run(x, return_pre=True)
st = running_stats(x)

# variant B: feedback from wave units approximated by (level eigenvector ratio) * h_2
ratio = np.zeros(16)
for u in WAVE:
    ratio[u] = RIGHT["level"][ORD.index(u)].real / RIGHT["level"][ORD.index(2)].real
fb = np.array([sum(W[j, u] * ratio[u] for u in WAVE if u != 2) for j in core])   # extra weight from unit 2
print("extra effective weight from unit 2 (variant B) into core units", dict(zip(core, np.round(fb, 3))))

def sim(variant):
    Wc = W[np.ix_(core, core)].copy()
    if variant == "B":
        Wc[:, core.index(2)] += fb
    h = np.zeros((len(x), len(core)))
    Hc = np.empty((len(x), T, len(core))); Zc = np.empty_like(Hc)
    for t in range(T):
        z = x[:, t, None] * w[core] + h @ Wc.T
        h = np.maximum(z, 0)
        Hc[:, t], Zc[:, t] = h, z
    return Hc, Zc

def yardstick():
    """No dynamics: thresholds theta_j(t) = -(W h_(t-1))_j / w_j replaced by c_j Mp(t), c_j = least-squares constant."""
    Mp = np.concatenate([np.zeros((len(x), 1)), np.maximum(st["M1"][:, :-1], 0)], 1)[:, :, None]   # max(0, x_0..x_(t-1))
    theta = -(Z[:, :, core] - x[:, :, None] * w[core]) / w[core]                                     # real thresholds (B, T, 6)
    c = (theta * Mp).sum((0, 1)) / (Mp ** 2).sum()
    print("\nyardstick: thresholds = c_j * running max of the earlier inputs, c_j =", {u: round(float(v), 3) for u, v in zip(core[1:], c[1:])},
          f"; unit 2 = ReLU(w_2 x_t + {-c[0] * w[2]:.1f} * running max)")
    Zy = w[core] * (x[:, :, None] - c * Mp)
    return np.maximum(Zy, 0), Zy

R2 = {}
for variant, model in (("A", lambda: sim("A")), ("B", lambda: sim("B")), ("yardstick (no dynamics)", yardstick)):
    Hc, Zc = model()
    print(f"\nvariant {variant}:")
    R2[variant] = []
    for k, u in enumerate(core):
        a, b = H[:, :, u].ravel(), Hc[:, :, k].ravel()
        r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        R2[variant].append(r2)
        print(f"   unit {u}: R2 of activation = {r2:.4f}   gate agreement = {np.mean((Z[:, :, u] > 0) == (Zc[:, :, k] > 0)):.4f}")
    f_real = Z[:, 1:, 4] > 0; f_sim = Zc[:, 1:, core.index(4)] > 0
    newmax = x[:, 1:] > st["M1"][:, :-1]
    print(f"   unit 4 fires: real {f_real.mean():.4f}, simulated {f_sim.mean():.4f}; "
          f"P(sim fires | new max) = {f_sim[newmax].mean():.4f} (real {f_real[newmax].mean():.4f}); "
          f"P(sim fires | not new max) = {f_sim[~newmax].mean():.4f} (real {f_real[~newmax].mean():.4f})")
    print("   R2 of unit 2 by step:", [round(1 - ((H[:, t, 2] - Hc[:, t, 0]) ** 2).sum() / ((H[:, t, 2] - H[:, t, 2].mean()) ** 2).sum(), 3) for t in range(1, 10)])

# NB activation R2 is lenient (see the yardstick row), and the functional test refuted the 'closed core' reading: cutting the satellites' weights into the core leaves 50.6% accuracy (notes/v_V4_core.md).
print("\ncentred R2 of activations over all steps, side by side (a lenient measure: compare with the yardstick);  units", core)
for variant, r in R2.items():
    print(f"   {variant:24s}" + "".join(f"{v:8.3f}" for v in r))
