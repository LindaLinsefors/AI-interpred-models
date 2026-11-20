"""C5c (diagnostic). Where does the clock's contribution to the final logits come from when nothing happens after step t?"""
from cl import *
NS, SEED = 100_000, 1502
x, H, Z, st = gen(NS, SEED)
M, S, a1, p1 = st["M1"], pos(st["M2"]), st["a1"], st["p1"]
G = (Z > 0).astype(float)
ff = np.zeros(16); ff[ORD] = 1                      # free-flow gates: ordinary on, pulses off
rows = []
for t in range(2, 9):
    xm = x[:, t + 1:].max(1)
    sel = (M[:, t] > 0) & (a1[:, t] >= 1) & (xm < S[:, t])       # class A
    idx = np.nonzero(sel)[0]
    zc = H[idx, t][:, ORD] @ LC
    v0 = np.zeros((idx.size, 16)); v0[:, ORD] = 2 * np.real(np.multiply.outer(zc, RC))
    variants = {}
    for name in ("actual", "free", "actual, last step free", "free, last step actual", "actual but pulses 5,9 gated off"):
        v = v0.copy()
        for tp in range(t + 1, 10):
            if name == "actual": g = G[idx, tp]
            elif name == "free": g = ff
            elif name == "actual, last step free": g = ff if tp == 9 else G[idx, tp]
            elif name == "free, last step actual": g = G[idx, tp] if tp == 9 else ff
            else:
                g = G[idx, tp].copy(); g[:, [5, 9]] = 0
            v = (v @ W.T) * g
        variants[name] = (v @ U.T)[np.arange(idx.size), p1[idx, t]]
    gl = G[idx, 9]
    rows.append((t, idx.size, {k: v.mean() for k, v in variants.items()}, 1 - gl[:, COMP].mean(0), gl[:, PULSE].mean(0),
                 np.mean([1 - G[idx, tp][:, COMP].mean(0) for tp in range(t + 1, 9)], 0) if t < 8 else np.zeros(4),
                 np.mean([G[idx, tp][:, PULSE].mean(0) for tp in range(t + 1, 9)], 0) if t < 8 else np.zeros(3), a1[idx, t]))
print("class A (nothing above S+_t arrives after t): mean contribution of the clock component of h_t at the max's own position, by gate variant")
for t, n, d, c9, pu9, cmid, pumid, _ in rows:
    print(f"  t={t} n={n}: " + "  ".join(f"{k}: {v:+.2f}" for k, v in d.items()))
    print(f"        final step: comparators clipped (1,6,7,8) {np.round(c9, 3)}  pulses on (4,5,9) {np.round(pu9, 3)} | steps t+1..8: clipped {np.round(cmid, 3)}  pulses on {np.round(pumid, 3)}")
# wave-unit gates in class A
t = 5
xm = x[:, t + 1:].max(1)
idx = np.nonzero((M[:, t] > 0) & (a1[:, t] >= 1) & (xm < S[:, t]))[0]
print("wave units on-share at steps 6..9 in class A (t=5):", np.round(np.mean([G[idx, tp][:, WAVE].mean(0) for tp in range(6, 10)], 0), 3), " units", WAVE)
