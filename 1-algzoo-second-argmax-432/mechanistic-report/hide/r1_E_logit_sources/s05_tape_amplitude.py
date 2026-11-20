"""S05. What sets the amplitude of the unit-5 pulse ('tape write')?  [EXACT decomposition + FIT]  seed 1100, n = 400k.
h5_{s+1} = ( w_5 x_{s+1} + W[5].h_s )_+ .  Write the argument as a function of x_s with the state before step s fixed."""
from src import *

x = sample(400_000, 1100)
H, Z = run(x, return_pre=True)
B = len(x)
rs = running_stats(x)
Hprev = np.concatenate([np.zeros((B, 1, 16)), H[:, :-1]], 1)
th = {j: -(Hprev @ W[j]) / w[j] for j in COMP + [4]}          # thresholds at each step
# 'tape threshold' theta_5(s): value of x_s at which unit 5's pre-activation at s+1 crosses zero, computed in the regime where
# comparators 7 and 8 are ON at step s (x_s below both thresholds): pre5(s+1) = 18.28 (x_s - theta_7/8 mix) + rest
s = np.arange(2, 9)                                            # steps s = 2..8 (so s+1 <= 9 and the prefix has >= 2 elements)
xs = x[:, s]; pre5 = Z[:, s + 1, 5]; h5 = H[:, s + 1, 5]
low = (xs < th[7][:, s]) & (xs < th[8][:, s])                  # both comparators 7, 8 on at step s
slope = -(W[5, 7] * abs(w[7]) + W[5, 8] * abs(w[8]))
# linear in x_s only approximately (other units also see x_s a little); empirical zero crossing by extrapolation:
theta5 = xs - pre5 / (W[5, ORD] @ w[ORD])
M1 = np.maximum(rs["M1"][:, s - 1], 0); M2 = np.maximum(rs["M2"][:, s - 1], 0); M3 = np.maximum(rs["M3"][:, s - 1], 0)
mk = low
X = np.stack([M1[mk], M2[mk], M3[mk], np.ones(mk.sum())], 1); y = theta5[mk]
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"regime: comparators 7 and 8 both on at step s (n = {mk.sum()}, {mk.mean():.3f} of step-sequence pairs, s = 2..8)")
print(f"[FIT] theta_5 := x_s - pre5(s+1)/18.01  ~  {beta[0]:.3f} M1+ {beta[1]:+.3f} M2+ {beta[2]:+.3f} M3+ {beta[3]:+.3f}   R^2 = {1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
print(f"   (M1+, M2+, M3+ = positive parts of the three largest of x_0..x_(s-1));  mean theta_5 - M2+ = {np.mean(y - M2[mk]):+.3f}, mean theta_5 - M3+ = {np.mean(y - M3[mk]):+.3f}")
print(f"   P(theta_5 < M2+) = {np.mean(y < M2[mk]):.3f},  P(theta_5 < M3+) = {np.mean(y < M3[mk]):.3f}")
# how linear is h5 in x_s?  slope check by finite difference on a subsample
sub = np.arange(20000)
for st in (4, 7):
    xp = x[sub].copy(); xp[:, st] += 1e-4
    Hp = run(xp)
    d = (Hp[:, st + 1, 5] - H[sub, st + 1, 5]) / 1e-4
    on = H[sub, st + 1, 5] > 0
    reg_low = (x[sub, st] < th[7][sub, st]) & (x[sub, st] < th[8][sub, st])
    reg_mid = (x[sub, st] > th[8][sub, st]) & (x[sub, st] < th[7][sub, st])
    reg_top = (x[sub, st] > th[4][sub, st]) & (x[sub, st] > th[7][sub, st]) & (x[sub, st] > th[8][sub, st])
    for nm, rg in (("x_s below theta_7, theta_8", reg_low), ("theta_8 < x_s < theta_7", reg_mid), ("x_s above theta_4, theta_7, theta_8", reg_top)):
        m = on & rg
        print(f"   step {st}: d h5_(s+1)/d x_s where unit 5 fires and {nm:36s}: median {np.median(d[m]):6.2f}, IQR {np.percentile(d[m], 25):6.2f}..{np.percentile(d[m], 75):6.2f}  (n = {m.sum()})")
# amplitude of the tape write by the rank of the input in its prefix
rank = np.stack([(x[:, : t + 1] > x[:, t:t + 1]).sum(1) + 1 for t in range(10)], 1)
print("\nmean h5_(s+1) and mean h4_s by rank of x_s in its prefix (s = 2..8), positive x_s only:")
for rk in (1, 2, 3, 4):
    m = (rank[:, s] == rk) & (xs > 0)
    print(f"   rank {rk}: P(h5>0) = {np.mean(h5[m] > 0):.3f}, mean h5 = {h5[m].mean():6.2f}; mean h4_s = {H[:, s, 4][m].mean():6.2f};  mean x_s = {xs[m].mean():.2f}")
# the record case: h5_(s+1) vs h4_s
m = (rank[:, s] == 1) & (H[:, s, 4] > 0)
X = np.stack([H[:, s, 4][m], M1[m], np.ones(m.sum())], 1); y = h5[m]
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"[FIT] records: h5_(s+1) ~ {beta[0]:.3f} h4_s {beta[1]:+.3f} M1+(before) {beta[2]:+.3f}   R^2 = {1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}  (n = {m.sum()})")
