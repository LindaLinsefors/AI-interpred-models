"""Where is the threshold of unit 5?  Unit 5 at step t+1 has pre-activation
     z_5(t+1) = w_5 x_{t+1} + W[5].h_t ,   h_t = D_t (w x_t + W h_{t-1})   (D_t = gates at step t)
so, with the gates of step t held fixed,
     z_5(t+1) = g_t x_t + W[5] D_t W h_{t-1} + w_5 x_{t+1},     g_t = W[5] D_t w ,
and unit 5 fires at t+1 iff  x_t > theta_5(t) := x_t - z_5(t+1)/g_t   (for g_t > 0).
g_t = 18.0 when comparators 7 and 8 are on at step t (they carry the gain: W[5,7] w_7 + W[5,8] w_8 = 18.2).
theta_5 is reported relative to the running max M and runner-up S of x_0..x_{t-1} (positive parts), for the states in
which comparators 7 and 8 are on and unit 4 is off at step t (x_t is below those three thresholds)."""
from icommon import *

print(f"[EXACT] W[5,ORD].w[ORD] = {W[5, ORD] @ w[ORD]:.3f};  W[5,7] w_7 + W[5,8] w_8 = {W[5, 7] * w[7] + W[5, 8] * w[8]:.3f};  w_5 = {w[5]:.3f}")
x = sample(400000, 1041)
B = len(x)
H, Z = run(x, return_pre=True)
G = (Z > 0)
rs = running_stats(x)
sh = lambda a, fill: np.concatenate([np.full((B, 1), fill), a[:, :-1]], 1)
M = sh(rs["M1"], -np.inf); S = sh(rs["M2"], -np.inf)
tt = np.broadcast_to(np.arange(T), (B, T))
aM = tt - sh(rs["p1"], -1); aS = tt - sh(rs["p2"], -1)
g = (G * (W[5] * w)).sum(-1)                                             # g_t = W[5] D_t w
z5next = np.concatenate([Z[:, 1:, 5], np.zeros((B, 1))], 1)
with np.errstate(divide="ignore", invalid="ignore"):
    theta5 = x - z5next / g
allord = G[..., ORD].all(-1) & ~G[..., PULSE].any(-1)
print(f"share of states (t = 1..8) with all 13 ordinary units on and pulses off: {allord[:, 1:9].mean():.3f};  with comparators 7, 8 on and unit 4 off: {(G[..., 7] & G[..., 8] & ~G[..., 4])[:, 1:9].mean():.3f}")
ok = G[..., 7] & G[..., 8] & ~G[..., 4] & (tt <= 8) & (tt >= 1)
print(f"gain g_t in those states: quantiles 1/50/99% = " + " / ".join(f"{v:.2f}" for v in np.quantile(g[ok], [.01, .5, .99])))
fires_next = z5next > 0
print(f"check: unit 5 fires at t+1 iff x_t > theta_5(t): {np.mean((x > theta5)[ok] == fires_next[ok]):.6f} of {ok.sum()} states")

Sp = np.maximum(S, 0.0)
sel = ok & (M > 0)
lam = (theta5 - Sp) / M
print(f"\nstates with M > 0 (n = {sel.sum()}): (theta_5 - S+)/M quantiles 5/25/50/75/95%: " + " / ".join(f"{v:+.3f}" for v in np.quantile(lam[sel], [.05, .25, .5, .75, .95])))
print("by age of the max at the step: median (theta_5 - S+)/M without a positive runner-up (S <= 0) | with one (S > 0);  [FIT] theta_5/M = a + b S+/M over S > 0")
rows = []
for a in range(1, 9):
    s0 = sel & (aM == a) & (S <= 0); s1 = sel & (aM == a) & (S > 0)
    X = np.c_[np.ones(s1.sum()), Sp[s1] / M[s1]]; y = theta5[s1] / M[s1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rows.append([a, s0.sum(), f"{np.median(lam[s0]):+.3f}", s1.sum(), f"{np.median(lam[s1]):+.3f}",
                 f"{np.subtract(*np.quantile(lam[s1], [.75, .25])):.3f}", f"{beta[0]:+.3f}", f"{beta[1]:.3f}", f"{r2:.3f}"])
print(fmt_table(rows, ["age of max", "n (S<=0)", "median, S <= 0", "n (S>0)", "median, S > 0", "IQR, S > 0", "a", "b", "R^2"]))
print("\nby age of the runner-up at the step (S > 0): median (theta_5 - S)/M")
print(fmt_table([[b, (sel & (S > 0) & (aS == b)).sum(), f"{np.median(lam[sel & (S > 0) & (aS == b)]):+.3f}"] for b in range(1, 9)], ["age of runner-up", "n", "median"]))
s1 = sel & (S > 0)
print(f"\nwith S > 0: P(theta_5 > S) = {np.mean(theta5[s1] > S[s1]):.4f};  P(theta_5 < S - 0.1 M) = {np.mean(theta5[s1] < S[s1] - 0.1 * M[s1]):.4f};  P(theta_5 < S - 0.2 M) = {np.mean(theta5[s1] < S[s1] - 0.2 * M[s1]):.4f}")
