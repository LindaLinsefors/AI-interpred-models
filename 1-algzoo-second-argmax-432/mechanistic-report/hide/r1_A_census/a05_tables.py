"""A5 (second part): tables A5.4 - A5.7 from the threshold positions; called by a05_thresholds.py.
pos[j] has columns: u, a1 (age of max in h_(t-1)), a2 (age of runner-up in h_(t-1)), t, theta, M, S."""
from cc import *

UN = [8, 6, 1, 7, 4, 5, 9]


def main(pos):
    print("\n## A5.4 median position by (age of max a1, age of runner-up a2) in h_(t-1); cells with n >= 2000\n")
    for j in UN:
        print(f"\nunit {j}: rows a1 = 0..8, columns a2 = 0..8\n")
        rows = []
        for a in range(9):
            r = [a]
            for b in range(9):
                m = (pos[j][:, 1] == a) & (pos[j][:, 2] == b)
                r.append(f"{np.median(pos[j][m, 0]):+.2f}" if m.sum() >= 2000 else ".")
            rows.append(r)
        print(md_table(["a1 \ a2"] + [str(b) for b in range(9)], rows))

    print("\n## A5.5 [FIT] theta = alpha*M + beta*S (least squares, no intercept)\n")
    print("(a) one fit per age of the max a1 (pooled over the age of the runner-up): alpha / beta / R^2\n")
    rows = []
    for a in range(9):
        r = [a]
        for j in UN:
            m = pos[j][:, 1] == a
            if m.sum() < 2000:
                r.append("-"); continue
            X = pos[j][m][:, [5, 6]]; y = pos[j][m, 4]
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            r.append(f"{coef[0]:+.3f} / {coef[1]:+.3f} / {r2:.3f}")
        rows.append(r)
    print(md_table(["a1"] + [f"unit {j}" for j in UN], rows))
    print("\n(b) one fit per (a1, a2) cell with n >= 2000, pooled over cells\n")
    rows = []
    for j in UN:
        dd = pos[j]
        sse = sst = 0.0; ncell = 0; npts = 0
        for a in range(9):
            for b in range(9):
                m = (dd[:, 1] == a) & (dd[:, 2] == b)
                if m.sum() < 2000:
                    continue
                X = dd[m][:, [5, 6]]; y = dd[m][:, 4]
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse += ((y - X @ coef) ** 2).sum(); sst += ((y - y.mean()) ** 2).sum(); ncell += 1; npts += m.sum()
        rows.append([j, ncell, npts, f"{1 - sse / sst:.4f}", f"{np.sqrt(sse / npts):.3f}"])
    print(md_table(["unit", "cells", "n", "pooled R^2 (share of within-cell variance of theta explained)", "rms residual (input units)"], rows))

    print("\n## A5.6 share of steps where the threshold lies outside [S, M] (same selection)\n")
    rows = [[j, f"{(pos[j][:, 0] < 0).mean():.4f}", f"{(pos[j][:, 0] > 1).mean():.4f}"] for j in UN]
    print(md_table(["unit", "P(theta < S)", "P(theta > M)"], rows))

    print("\n## A5.7 delayed threshold of unit 5 as a fraction of S: theta_5 / S, quartiles 25% / 50% / 75%\n")
    q = lambda a: " / ".join(f"{v:.3f}" for v in np.percentile(a, [25, 50, 75])) if len(a) else "-"
    dd = pos[5]
    rows = [[a, int((dd[:, 1] == a).sum()), q((dd[:, 4] / dd[:, 6])[dd[:, 1] == a])] for a in range(8)]
    rows.append(["all", len(dd), q(dd[:, 4] / dd[:, 6])])
    print(md_table(["age of max in h_(t-1)", "n", "theta_5 / S"], rows))
    print("\nby S/M bin (all ages): theta_5 / S quartiles\n")
    sm = dd[:, 6] / dd[:, 5]
    rows = [[f"[{lo:.1f}, {hi:.1f})", int(((sm >= lo) & (sm < hi)).sum()), q((dd[:, 4] / dd[:, 6])[(sm >= lo) & (sm < hi)])]
            for lo, hi in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.01, 0.1))]
    print(md_table(["S/M", "n", "theta_5 / S"], rows))
