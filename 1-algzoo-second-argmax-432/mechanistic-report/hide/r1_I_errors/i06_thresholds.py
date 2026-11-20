"""Where do the thresholds of the comparators (1, 6, 7, 8) and of unit 4 sit, relative to the running max M
and running runner-up S (positive parts), as a function of the ages of both at the step the threshold is used?

  theta_j(t) = -W[j].h_{t-1}/w_j ;  lambda_j = (theta_j - S+)/(M - S+)  (0: at the runner-up, 1: at the max)
  M, S = largest and second largest of x_0..x_{t-1};  S+ = max(S, 0);  ages counted at step t: t - p.
"""
from icommon import *

x = sample(400000, 1040)
B = len(x)
H, Zp = run(x, return_pre=True)
rs = running_stats(x)
Hprev = np.concatenate([np.zeros((B, 1, N)), H[:, :-1]], 1)
sh = lambda a, fill: np.concatenate([np.full((B, 1), fill), a[:, :-1]], 1)
M = sh(rs["M1"], -np.inf); S = sh(rs["M2"], -np.inf); Tv = sh(rs["M3"], -np.inf)
tt = np.broadcast_to(np.arange(T), (B, T))
aM = tt - sh(rs["p1"], -1); aS = tt - sh(rs["p2"], -1)
Sp = np.maximum(S, 0.0); Tp = np.maximum(Tv, 0.0)
valid = (tt >= 2) & (M > 0)

for j in [4] + COMP:
    theta = -(Hprev @ W[j]) / w[j]
    lam = (theta - Sp) / (M - Sp)
    print(f"\n## unit {j} (w = {w[j]:.2f}): median lambda = (theta - S+)/(M - S+) by age of max (rows) and of runner-up (columns) at the step; S > 0 only")
    rows = []
    for a in range(1, 10):
        row = [a]
        for b in range(1, 10):
            sel = valid & (S > 0) & (aM == a) & (aS == b)
            row.append("." if sel.sum() < 300 else f"{np.median(lam[sel]):.2f}")
        rows.append(row)
    print(fmt_table(rows, ["aM \\ aS"] + [str(b) for b in range(1, 10)]))
    print(f"[FIT] theta/M = alpha + beta * S+/M + gamma * T+/M within each (aM, aS) cell; cells with n >= 300; summary over cells")
    al = []; be = []; ga = []; r2 = []; cells = []
    for a in range(1, 10):
        for b in range(1, 10):
            sel = valid & (aM == a) & (aS == b)
            if sel.sum() < 300: continue
            y = theta[sel] / M[sel]
            X = np.c_[np.ones(sel.sum()), Sp[sel] / M[sel], Tp[sel] / M[sel]]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            al.append(beta[0]); be.append(beta[1]); ga.append(beta[2]); cells.append((a, b))
            r2.append(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    al, be, ga, r2 = map(np.array, (al, be, ga, r2))
    print(f"  alpha: min {al.min():.2f} median {np.median(al):.2f} max {al.max():.2f};  beta: min {be.min():.2f} median {np.median(be):.2f} max {be.max():.2f};"
          f"  gamma: min {ga.min():.2f} median {np.median(ga):.2f} max {ga.max():.2f};  R^2: min {r2.min():.3f} median {np.median(r2):.3f}")
    print("  alpha table (theta/M when S+ = T+ = 0):")
    rows = []
    for a in range(1, 10):
        row = [a]
        for b in range(1, 10):
            row.append(f"{al[cells.index((a, b))]:.2f}" if (a, b) in cells else ".")
        rows.append(row)
    print(fmt_table(rows, ["aM \\ aS"] + [str(b) for b in range(1, 10)]))
    print("  alpha + beta table (theta/M when S+ = M, T+ = 0):")
    rows = []
    for a in range(1, 10):
        row = [a]
        for b in range(1, 10):
            row.append(f"{al[cells.index((a, b))] + be[cells.index((a, b))]:.2f}" if (a, b) in cells else ".")
        rows.append(row)
    print(fmt_table(rows, ["aM \\ aS"] + [str(b) for b in range(1, 10)]))

# With one positive input only (S <= 0): thresholds in units of M, by age of the max
print("\n## only one positive input so far (M > 0 >= S): median theta_j / M by age of the max")
rows = []
for a in range(1, 10):
    sel = (tt >= 1) & (M > 0) & (S <= 0) & (aM == a)
    row = [a, sel.sum()]
    for j in [4] + COMP:
        theta = -(Hprev @ W[j]) / w[j]
        row.append(f"{np.median(theta[sel] / M[sel]):.3f}")
    rows.append(row)
print(fmt_table(rows, ["age of max", "n", "unit 4", "unit 1", "unit 6", "unit 7", "unit 8"]))
print("\n## no positive input so far (M <= 0): theta_j quantiles 1% / 50% / 99% (absolute units)")
sel = (tt >= 1) & (M <= 0)
for j in [4] + COMP:
    theta = -(Hprev @ W[j]) / w[j]
    print(f"  unit {j}: " + " / ".join(f"{v:.4f}" for v in np.quantile(theta[sel], [.01, .5, .99])), f"  P(theta > 0) = {np.mean(theta[sel] > 0):.3f}")
