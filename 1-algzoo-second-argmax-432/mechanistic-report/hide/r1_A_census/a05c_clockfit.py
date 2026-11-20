"""A5 extra [FIT]: do the same-step threshold positions u_j = (theta_j - S)/(M - S) oscillate at the clock frequency?

Fit  u_j(a) = c_j + r_j cos(omega a - phi_j)  with omega = 42.75 deg (angle of the clock eigenvalue of A = W[ORD, ORD]),
a = age of the max in h_(t-1), on all steps with S > 0, M - S > 0.2 and a >= a_min.  Because a takes only 9 values the
least-squares fit is done exactly from the per-age counts, sums and sums of squares.  Also: frequency left free (grid)."""
from cc import *

om0 = np.degrees(np.angle(LAM["clock"]))
UN = [8, 6, 1, 7, 4]
st = {j: np.zeros((9, 3)) for j in UN}          # per age: n, sum u, sum u^2
for seed, x, H, Z in batches():
    ev = events(x); B = len(x)
    Hprev = np.concatenate([np.zeros((B, 1, N)), H[:, :-1]], 1)
    Mp, Sp = ev["Mp"], ev["Sp"]
    a1p = np.concatenate([np.full((B, 1), -1), ev["a1"][:, :-1]], 1)
    with np.errstate(invalid="ignore"):
        sel = (Sp > 0) & (Mp - Sp > 0.2)
    sel[:, 0] = False
    for j in UN:
        th = -(Hprev @ W[j]) / w[j]
        u = (th[sel] - Sp[sel]) / (Mp[sel] - Sp[sel]); a = a1p[sel]
        for k in range(9):
            v = u[a == k]
            st[j][k] += [len(v), v.sum(), (v ** 2).sum()]
print(f"clock angle omega = {om0:.2f} deg\n")
for amin in (2, 0):
    rows = []
    for j in UN:
        n, s1, s2 = st[j][amin:].T
        ages = np.arange(amin, 9); mean = s1 / n
        tot_mean = s1.sum() / n.sum(); sst = s2.sum() - n.sum() * tot_mean ** 2
        within = (s2 - n * mean ** 2).sum()

        def fit(om):
            X = np.stack([np.ones(len(ages)), np.cos(np.radians(om) * ages), np.sin(np.radians(om) * ages)], 1)
            Wt = np.sqrt(n)[:, None]
            coef, *_ = np.linalg.lstsq(X * Wt, mean * Wt[:, 0], rcond=None)
            sse = within + (n * (mean - X @ coef) ** 2).sum()
            return coef, 1 - sse / sst
        coef, r2 = fit(om0)
        grid = np.arange(20.0, 90.01, 0.25)
        r2s = [fit(om)[1] for om in grid]
        rows.append([j, int(n.sum()), f"{coef[0]:.3f}", f"{np.hypot(coef[1], coef[2]):.3f}", f"{np.degrees(np.arctan2(coef[2], coef[1])) % 360:.1f}",
                     f"{r2:.4f}", f"{1 - within / sst:.4f}", f"{grid[int(np.argmax(r2s))]:.2f}", f"{max(r2s):.4f}"])
    print(f"ages a >= {amin}\n")
    print(md_table(["unit", "n", "c (mean position)", "r (amplitude)", "phi (deg)", "R^2 at omega = clock",
                    "R^2 ceiling (any function of a)", "best free omega (deg)", "R^2 at best omega"], rows))
    print()
