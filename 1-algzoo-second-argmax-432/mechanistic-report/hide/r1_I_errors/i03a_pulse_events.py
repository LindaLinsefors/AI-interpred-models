"""I3 (preparation). When do the pulse units 4, 5, 9 fire on N(0,1) data, relative to the events
'x_t is a new max' / 'x_t is above the runner-up so far'?  Threshold of unit 4 against the running max."""
from icommon import *

x = sample(400000, 1020)
B = len(x)
H, Z = run(x, return_pre=True)
rs = running_stats(x)
ok = logits(H).argmax(1) == second_argmax(x)
Hprev = np.concatenate([np.zeros((B, 1, N)), H[:, :-1]], 1)

# exclusive prefix statistics (x_0..x_{t-1}); -inf when absent
Mprev = np.concatenate([np.full((B, 1), -np.inf), rs["M1"][:, :-1]], 1)
Sprev = np.concatenate([np.full((B, 1), -np.inf), rs["M2"][:, :-1]], 1)
Tprev = np.concatenate([np.full((B, 1), -np.inf), rs["M3"][:, :-1]], 1)
ref = np.maximum(Mprev, 0.0)                      # max(M_{t-1}, 0)
oracle = W4 * np.maximum(x - ref, 0.0)
h4 = H[..., 4]
on4 = h4 > 0
rec = x > ref                                     # oracle fires
newmax = x > Mprev                                # x_t is a new max (possibly negative)

print("## unit 4 against the oracle 10.157 * max(x_t - max(M_{t-1}, 0), 0)   (n = %d sequences x 10 steps)" % B)
print(f"R^2 of h4 against the oracle over all states: {1 - ((h4 - oracle)**2).sum() / ((h4 - h4.mean())**2).sum():.5f}")
print(f"states: record & on {np.mean(rec & on4):.5f}, record & silent (miss) {np.mean(rec & ~on4):.5f}, "
      f"no record & on (misfire) {np.mean(~rec & on4):.5f}, neither {np.mean(~rec & ~on4):.5f}")
print(f"P(on | record) = {on4[rec].mean():.5f};  P(on | no record) = {on4[~rec].mean():.5f}")
print(f"new max but negative (x_t > M_(t-1), x_t <= 0): share {np.mean(newmax & ~rec):.5f}, unit 4 on in {on4[newmax & ~rec].mean():.5f}")
print(f"sequences with >= 1 misfire: {np.mean((~rec & on4).any(1)):.4f}; with >= 1 miss: {np.mean((rec & ~on4).any(1)):.4f}")
print(f"accuracy: no misfire & no miss {ok[~((~rec & on4).any(1)) & ~((rec & ~on4).any(1))].mean():.4f}; "
      f">=1 misfire {ok[(~rec & on4).any(1)].mean():.4f}; >=1 miss {ok[(rec & ~on4).any(1)].mean():.4f}")

mf = ~rec & on4
print("\nmisfires: activation h4 quantiles (5, 25, 50, 75, 95, 99%):", np.round(np.quantile(h4[mf], [.05, .25, .5, .75, .95, .99]), 3))
print("misfires: shortfall max(M_{t-1},0) - x_t quantiles:", np.round(np.quantile((ref - x)[mf], [.05, .25, .5, .75, .95, .99]), 3))
print("misfires: relative shortfall (ref - x_t)/ref quantiles (ref > 0):", np.round(np.quantile(((ref - x) / ref)[mf & (ref > 0)], [.05, .25, .5, .75, .95, .99]), 3))
print("misfires with M_{t-1} <= 0 (reference is 0):", np.mean(Mprev[mf] <= 0))
ms = rec & ~on4
print("misses: excess x_t - max(M_{t-1},0) quantiles:", np.round(np.quantile((x - ref)[ms], [.05, .25, .5, .75, .95, .99]), 3))
print("misses: relative excess (x_t - ref)/x_t quantiles:", np.round(np.quantile(((x - ref) / x)[ms], [.05, .25, .5, .75, .95, .99]), 3))

# threshold of unit 4 against the running max, by age of the max at t-1 (steps since it arrived, counted at t)
theta4 = -(Hprev @ W[4]) / w[4]
tt = np.broadcast_to(np.arange(T), (B, T))
p1prev = np.concatenate([np.full((B, 1), -1), rs["p1"][:, :-1]], 1)
age = tt - p1prev                                   # age of the previous max at step t (>= 1)
print("\n## threshold of unit 4: theta_4(t)/M_{t-1} for M_{t-1} > 0, by age of that max at step t (t - p1)")
rows = []
for a in range(1, 10):
    sel = (Mprev > 0) & (age == a) & (tt >= 1)
    r = theta4[sel] / Mprev[sel]
    q = np.quantile(r, [.01, .1, .5, .9, .99])
    rows.append([a, sel.sum(), f"{r.mean():.4f}", f"{r.std():.4f}"] + [f"{v:.3f}" for v in q] +
                [f"{np.mean(mf[sel]):.4f}", f"{np.mean(ms[sel]):.4f}"])
print(fmt_table(rows, ["age of max", "n", "mean theta4/M", "sd", "q1%", "q10%", "q50%", "q90%", "q99%", "misfire rate", "miss rate"]))
# what explains the deviation: regress (theta4 - M)/M on S+/M and T+/M within age
print("\n[FIT] (theta_4 - M)/M ~ a + b*(S+/M) + c*(T+/M) within age cells (M = M_{t-1} > 0; S+, T+ = positive parts of 2nd, 3rd so far)")
rows = []
for a in range(1, 10):
    sel = (Mprev > 0) & (age == a) & (tt >= 1)
    y = (theta4[sel] - Mprev[sel]) / Mprev[sel]
    Sp = np.maximum(Sprev[sel], 0) / Mprev[sel]; Tp = np.maximum(Tprev[sel], 0) / Mprev[sel]
    X = np.c_[np.ones(len(y)), Sp, Tp]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta)**2).sum() / ((y - y.mean())**2).sum()
    rows.append([a, f"{beta[0]:.4f}", f"{beta[1]:.4f}", f"{beta[2]:.4f}", f"{r2:.3f}"])
print(fmt_table(rows, ["age", "a", "b (S+/M)", "c (T+/M)", "R^2"]))

# states before any positive input: threshold relative to 0
sel = (Mprev <= 0) & (tt >= 1)
print("\n## no positive input yet (M_{t-1} <= 0): theta_4(t) quantiles (1, 10, 50, 90, 99%):", np.round(np.quantile(theta4[sel], [.01, .1, .5, .9, .99]), 4),
      " in units of |x_(t-1)|:", np.round(np.quantile((theta4 / np.abs(np.roll(x, 1, axis=1)))[sel], [.01, .1, .5, .9, .99]), 4))

# ---- unit 5 and 9 at t+1 against the rank of x_t in the prefix x_0..x_t
rank_t = (x[:, None, :] > x[:, :, None]).astype(np.int8)          # [b, t, s] = x_s > x_t
mask = np.tril(np.ones((T, T), dtype=np.int8), -1)               # s < t
rk = (rank_t * mask[None]).sum(2) + 1                              # rank of x_t among x_0..x_t (1 = new max)
for u in (5, 9):
    hu_next = H[:, 1:, u]                                          # unit u at t+1, t = 0..8
    print(f"\n## unit {u} at step t+1 against the rank of x_t among x_0..x_t (t = 0..8); and the sign of x_t")
    rows = []
    for name, sel in (("rank 1 (new max), x_t>0", (rk[:, :-1] == 1) & (x[:, :-1] > 0)), ("rank 1, x_t<=0", (rk[:, :-1] == 1) & (x[:, :-1] <= 0)),
                      ("rank 2, x_t>0", (rk[:, :-1] == 2) & (x[:, :-1] > 0)), ("rank 2, x_t<=0", (rk[:, :-1] == 2) & (x[:, :-1] <= 0)),
                      ("rank 3, x_t>0", (rk[:, :-1] == 3) & (x[:, :-1] > 0)), ("rank 3, x_t<=0", (rk[:, :-1] == 3) & (x[:, :-1] <= 0)),
                      ("rank 4+, x_t>0", (rk[:, :-1] >= 4) & (x[:, :-1] > 0)), ("rank 4+, x_t<=0", (rk[:, :-1] >= 4) & (x[:, :-1] <= 0))):
        v = hu_next[sel]
        rows.append([name, sel.sum(), f"{(v > 0).mean():.4f}", f"{v.mean():.3f}", f"{v[v > 0].mean() if (v > 0).any() else 0:.3f}"])
    print(fmt_table(rows, ["x_t", "n", f"P(unit {u} on at t+1)", "mean activation", "mean when on"]))

# unit 5: which threshold?  on5 at t+1 iff x_t > theta_5(t) with theta_5 defined by linearity in x_t given h_{t-1} ... only if gates at t fixed.
# Simple empirical statement: P(on5 at t+1) against (x_t - S_t-1)/M for records excluded
Sref = np.maximum(Sprev, 0.0)
d = ((x - Sref) / np.maximum(ref, 1e-9))[:, 1:-1]                   # t = 1..8
on5n = H[:, 2:, 5] > 0
sel0 = (Mprev[:, 1:-1] > 0) & ~rec[:, 1:-1]
print("\n## unit 5 at t+1 against u = (x_t - max(S_{t-1},0)) / M_{t-1}, x_t not a record, M_{t-1} > 0 (t = 1..8)")
edges = [-np.inf, -1, -0.5, -0.3, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
rows = []
for i in range(len(edges) - 1):
    sel = sel0 & (d >= edges[i]) & (d < edges[i + 1])
    if sel.sum() == 0: continue
    rows.append([f"[{edges[i]}, {edges[i+1]})", sel.sum(), f"{on5n[sel].mean():.4f}", f"{H[:, 2:, 5][sel].mean():.3f}", f"{(H[:, 2:, 9][sel] > 0).mean():.4f}", f"{H[:, 2:, 9][sel].mean():.3f}"])
print(fmt_table(rows, ["u", "n", "P(unit 5 on at t+1)", "mean h5", "P(unit 9 on at t+1)", "mean h9"]))
