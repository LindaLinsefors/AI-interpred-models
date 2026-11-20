"""S01. Basic statistics of the event amplitudes on N(0,1) data (to know what the sources are before aggregating).
seed 1002, n = 400k."""
from src import *

x = sample(400_000, 1002)
H, Z, p, c = amplitudes(x)
rs = running_stats(x)
B = len(x)
Hprev = np.concatenate([np.zeros((B, 1, 16)), H[:, :-1]], 1)

print("fraction of (sequence, step) with a nonzero amplitude, and mean amplitude when nonzero; steps 1..9 pooled")
print("source        P(>0)   mean|>0   by step s = 0..9: P(>0)")
for k, u in enumerate(PULSE):
    a = p[:, :, k]
    print(f"pulse {u:2d}     {(a[:, 1:] > 0).mean():.3f}  {a[:, 1:][a[:, 1:] > 0].mean():8.3f}   " + " ".join(f"{(a[:, s] > 0).mean():.3f}" for s in range(10)))
for i, u in enumerate(ORD):
    a = c[:, :, i]
    nz = a[:, 1:] > 0
    print(f"clip  {u:2d}     {nz.mean():.3f}  {a[:, 1:][nz].mean() if nz.any() else 0:8.3f}   " + " ".join(f"{(a[:, s] > 0).mean():.3f}" for s in range(10)))

# thresholds vs running statistics of the prefix x_0..x_{s-1}
print("\nthresholds theta_j(s) = -W[j].h_{s-1}/w_j regressed on [M1, max(M2,0)] of x_0..x_{s-1}  (s = 2..9 pooled, affine fit)")
M1 = rs["M1"][:, 1:-1].ravel(); M2 = np.maximum(rs["M2"][:, 1:-1], 0).ravel()   # prefix up to s-1, s = 2..9
M1p = np.maximum(M1, 0)
for j in COMP + [4]:
    th = (-(Hprev @ W[j]) / w[j])[:, 2:].ravel()
    X = np.stack([M1p, M2, np.ones_like(M1)], 1)
    beta, *_ = np.linalg.lstsq(X, th, rcond=None)
    r2 = 1 - ((th - X @ beta) ** 2).sum() / ((th - th.mean()) ** 2).sum()
    print(f"  unit {j}: theta ~ {beta[0]:+.3f} max(M1,0) {beta[1]:+.3f} max(M2,0) {beta[2]:+.3f}   R^2 = {r2:.4f}   "
          f"mean(theta - M1+) = {np.mean(th - M1p):+.3f}, mean(theta - M2+) = {np.mean(th - M2):+.3f}, P(theta < M2+) = {np.mean(th < M2):.3f}, P(theta > M1+) = {np.mean(th > M1p):.3f}")

# what triggers each event?  classify the input x_s by its rank in the prefix x_0..x_s
print("\nP(event at step s | rank of x_s among x_0..x_s), steps 2..9 pooled; rank 1 = new max, 2 = new runner-up, 3 = new third, 4+ = lower")
rank = np.stack([(x[:, : s + 1] > x[:, s:s + 1]).sum(1) + 1 for s in range(10)], 1)     # (B, 10)
pos_part = x > 0
hdr = "event                         rank1    rank2    rank3    rank4+ | rank1&x>0 rank2&x>0 rank3&x>0 rank4+&x>0"
print(hdr)


def by_rank(ev, label, shift=0):
    """ev (B,10) boolean at step s; attribute to the input at step s - shift."""
    r = rank[:, 2:10 - shift].ravel(); e = ev[:, 2 + shift:].ravel(); pp = pos_part[:, 2:10 - shift].ravel()
    row = []
    for cond in (r == 1, r == 2, r == 3, r >= 4):
        row.append(e[cond].mean())
    for cond in (r == 1, r == 2, r == 3, r >= 4):
        row.append(e[cond & pp].mean())
    print(f"{label:28s} " + " ".join(f"{v:8.3f}" for v in row))


by_rank(p[:, :, 0] > 0, "unit 4 fires at s")
by_rank(p[:, :, 1] > 0, "unit 5 fires at s+1", 1)
by_rank(p[:, :, 2] > 0, "unit 9 fires at s+1", 1)
for j in COMP:
    by_rank(c[:, :, ORD.index(j)] > 0, f"comparator {j} clipped at s")
for i in WAVE:
    by_rank(c[:, :, ORD.index(i)] > 0, f"wave {i} clipped at s")
    by_rank(c[:, :, ORD.index(i)] > 0, f"wave {i} clipped at s+1", 1)

# amplitude regressions for the pulses
print("\npulse amplitudes: [FIT] on steps where the pulse is on")
d1 = (x - np.maximum(np.concatenate([np.full((B, 1), 0.0), rs['M1'][:, :-1]], 1), 0))      # x_s - max(0, M1(s-1))
on = p[:, :, 0] > 0
X = np.stack([d1[on], np.ones(on.sum())], 1); y = p[:, :, 0][on]
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"  h4_s ~ {beta[0]:.3f} (x_s - max(0, M1_(s-1))) + {beta[1]:.3f}   R^2 = {1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum():.4f}   (n = {on.sum()})")
