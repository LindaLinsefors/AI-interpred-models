"""V6 claims D1 and D2: the lead of the old max over the old runner-up right after a new max.

D1: record at step t >= 2 (x_t > M_{t-1}), M0 = M_{t-1} > 0.3, S0 = S_{t-1} > 0.1, old max age
    g = t - 1 - pM0 >= 1.  y = s_t[old max] - s_t[old runner-up];  y ~ c1 (M0 - S0) + c2 M0 by
    Delta = |pM0 - pS0|.
D2: same records, top two (x_t, M0) unchanged for the next k steps; y read at t + k; pooled over Delta.
Writes d1_events.npz (cache for the attack scripts) and d1_d2.txt.
"""
from v6_data import *

out = open(HERE / "d1_d2.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


ev = {k: [] for k in ("seed", "t", "pM", "pS", "M0", "S0", "xt", "T0", "pT")}
yk = [[] for _ in range(6)]          # y read at t + k (nan if not valid)
ok2 = [[] for _ in range(6)]         # top two unchanged for k steps
ok3 = [[] for _ in range(6)]         # top three unchanged for k steps
for sd, x, H in batches():
    vals, pos = top3(x)
    for t in range(2, 10):
        M0, S0, T0 = vals[:, t - 1, 0], vals[:, t - 1, 1], vals[:, t - 1, 2]
        pM, pS, pT = pos[:, t - 1, 0], pos[:, t - 1, 1], pos[:, t - 1, 2]
        sel = (x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1) & (t - 1 - pM >= 1)
        i = np.flatnonzero(sel)
        for k_, a in zip(("seed", "t", "pM", "pS", "M0", "S0", "xt", "T0", "pT"),
                         (np.full(len(i), sd), np.full(len(i), t), pM[i], pS[i], M0[i], S0[i], x[i, t], T0[i], pT[i])):
            ev[k_].append(a)
        for k in range(6):
            if t + k > 9:
                yk[k].append(np.full(len(i), np.nan))
                ok2[k].append(np.zeros(len(i), bool))
                ok3[k].append(np.zeros(len(i), bool))
                continue
            Hi = H[i]
            y = score_at(Hi, t + k, pM[i]) - score_at(Hi, t + k, pS[i])
            later = x[i, t + 1: t + k + 1]
            u2 = (later < M0[i, None]).all(1)                 # nothing beats the old max
            u3 = (later < S0[i, None]).all(1)                 # nothing beats the old runner-up either
            yk[k].append(y)
            ok2[k].append(u2)
            ok3[k].append(u3)
ev = {k: np.concatenate(v) for k, v in ev.items()}
yk = np.stack([np.concatenate(v) for v in yk], 1)
ok2 = np.stack([np.concatenate(v) for v in ok2], 1)
ok3 = np.stack([np.concatenate(v) for v in ok3], 1)
np.savez_compressed(HERE / "d1_events.npz", yk=yk, ok2=ok2, ok3=ok3, **ev)

y = yk[:, 0]
gap = ev["M0"] - ev["S0"]
D = np.abs(ev["pM"] - ev["pS"])
P(f"seeds {SEEDS[0]}..{SEEDS[-1]}, {NB} sequences each; events: {len(y)}")
P("\nD1. y ~ c1 (M0 - S0) + c2 M0, no intercept, by Delta")
P("Delta       n     c1     c2  R2cen  R2unc  P(y>0)  c1/(1-cos)   | with intercept: c1    c2    c0  R2cen | c1 s.e. over seeds")
for d in range(1, 9):
    m = D == d
    b, r2c, r2u = ols([gap[m], ev["M0"][m]], y[m])
    bi, r2ci, _ = ols([gap[m], ev["M0"][m]], y[m], intercept=True)
    per = np.array([ols([gap[m & (ev["seed"] == s)], ev["M0"][m & (ev["seed"] == s)]], y[m & (ev["seed"] == s)])[0]
                    for s in SEEDS])
    se = per.std(0, ddof=1) / np.sqrt(len(SEEDS))
    P(f"{d:5d} {m.sum():7d} {b[0]:6.1f} {b[1]:6.2f} {r2c:6.3f} {r2u:6.3f} {np.mean(y[m] > 0):7.3f} "
      f"{b[0] / (1 - np.cos(OMEGA * d)):10.1f}   | {bi[0]:6.1f} {bi[1]:6.2f} {bi[2]:6.2f} {r2ci:6.3f} | "
      f"{se[0]:.2f} (c2 {se[1]:.2f})")
P("lead: c1 = 17.6, 43.5, 73.8, 88.5, 81.6, 63.2, 41.8; c2 = 0.58, 2.00, 5.03, 5.66, 4.34, 0.43, 2.05;")
P("      R2 = 0.65, 0.72, 0.67, 0.67, 0.73, 0.81, 0.36; P(y>0) = 0.960, 0.985, 0.979, 0.989, 0.968, 0.946, 0.893;")
P("      c1/(1-cos) = 65, 47, 46, 45, 45, 51, 82")

P("\nD2. y read at t + k, pooled over Delta; reading A = top two (x_t, M0) unchanged, reading B = top three unchanged")
for label, okk, dmax in (("A, Delta 1..8", ok2, 8), ("A, Delta 1..7", ok2, 7), ("B, Delta 1..8", ok3, 8), ("B, Delta 1..7", ok3, 7)):
    row, rowc2, rown, rowr = [], [], [], []
    for k in range(6):
        m = okk[:, k] & (D <= dmax)
        b, r2c, _ = ols([gap[m], ev["M0"][m]], yk[m, k])
        row.append(b[0]); rowc2.append(b[1]); rown.append(m.sum()); rowr.append(r2c)
    P(f"  {label}: c1 = " + ", ".join(f"{v:.1f}" for v in row) + " | c2 = " + ", ".join(f"{v:.2f}" for v in rowc2)
      + " | R2cen = " + ", ".join(f"{v:.2f}" for v in rowr) + " | n = " + ", ".join(str(v) for v in rown))
P("  lead: c1 = 40.7, 42.6, 35.3, 32.1, 28.6, 23.8 for k = 0..5")
# s.e. of the pooled c1 (reading A, Delta 1..8) over seeds
per = np.array([[ols([gap[m], ev["M0"][m]], yk[m, k])[0][0]
                 for k in range(6) for m in [ok2[:, k] & (ev["seed"] == s)]] for s in SEEDS])
P("  s.e. over seeds of c1 (reading A): " + ", ".join(f"{v:.2f}" for v in per.std(0, ddof=1) / np.sqrt(len(SEEDS))))
out.close()
