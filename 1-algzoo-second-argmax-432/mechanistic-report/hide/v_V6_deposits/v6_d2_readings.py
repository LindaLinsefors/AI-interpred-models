"""V6, D2 under every reading of the selection, plus a replay of D1/D2 on the lead's own sample
(600k sequences, seed 1414; read from the lead's e14 after my own numbers existed) to separate
sampling noise from definitional differences.

Readings of 'such records with the top two unchanged for the next k steps':
  A   g >= 1 (as D1 says), nothing in x_{t+1..t+k} beats the old max M0
  A0  same but g >= 0 (old max may sit at t-1)            <- what the lead's e14 actually does
  B   g >= 1, nothing beats the old runner-up S0 either (top three unchanged)
"""
from v6_data import *

out = open(HERE / "d2_readings.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


def collect(seeds, n):
    rows = {k: [] for k in ("seed", "t", "k", "g", "D", "gap", "M0", "y", "u2", "u3")}
    for sd in seeds:
        x = sample(n, sd)
        H = run(x)
        vals, pos = top3(x)
        for t in range(2, 10):
            M0, S0 = vals[:, t - 1, 0], vals[:, t - 1, 1]
            pM, pS = pos[:, t - 1, 0], pos[:, t - 1, 1]
            i = np.flatnonzero((x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1))
            Hi = H[i]
            for k in range(0, 10 - t):
                if k > 5:
                    break
                y = score_at(Hi, t + k, pM[i]) - score_at(Hi, t + k, pS[i])
                later = x[i, t + 1: t + k + 1]
                for k_, a in zip(("seed", "t", "k", "g", "D", "gap", "M0", "y", "u2", "u3"),
                                 (np.full(len(i), sd), np.full(len(i), t), np.full(len(i), k), t - 1 - pM[i],
                                  np.abs(pM[i] - pS[i]), M0[i] - S0[i], M0[i], y,
                                  (later < M0[i, None]).all(1), (later < S0[i, None]).all(1))):
                    rows[k_].append(a)
    return {k: np.concatenate(v) for k, v in rows.items()}


def table(r, title):
    P(title)
    for name, m0 in (("A  (g>=1, top two unchanged)", (r["g"] >= 1) & r["u2"]),
                     ("A0 (g>=0, top two unchanged)", r["u2"]),
                     ("B  (g>=1, top three unchanged)", (r["g"] >= 1) & r["u3"]),
                     ("B0 (g>=0, top three unchanged)", r["u3"])):
        c1, c2, r2, py, ns = [], [], [], [], []
        for k in range(6):
            m = m0 & (r["k"] == k)
            b, r2c, _ = ols([r["gap"][m], r["M0"][m]], r["y"][m])
            c1.append(b[0]); c2.append(b[1]); r2.append(r2c); py.append(np.mean(r["y"][m] > 0)); ns.append(m.sum())
        P(f"  {name}: c1 = " + ", ".join(f"{v:.1f}" for v in c1) + " | c2 = " + ", ".join(f"{v:.2f}" for v in c2)
          + " | R2cen = " + ", ".join(f"{v:.3f}" for v in r2) + " | P(y>0) = " + ", ".join(f"{v:.3f}" for v in py)
          + " | n = " + ", ".join(str(v) for v in ns))
    P("  lead: c1 = 40.7, 42.6, 35.3, 32.1, 28.6, 23.8 for k = 0..5")


r = collect(SEEDS, NB)
table(r, f"my seeds {SEEDS[0]}..{SEEDS[-1]} x {NB}")
# seed-to-seed s.e. for reading A0
per = np.array([[ols([r["gap"][m], r["M0"][m]], r["y"][m])[0][0]
                 for k in range(6) for m in [r["u2"] & (r["k"] == k) & (r["seed"] == s)]] for s in SEEDS])
P("  A0: s.e. of c1 over my 10 seeds (4M sequences): " + ", ".join(f"{v:.2f}" for v in per.std(0, ddof=1) / np.sqrt(len(SEEDS))))
P("  A0: s.d. of c1 between single 400k batches:      " + ", ".join(f"{v:.2f}" for v in per.std(0, ddof=1)))

rl = collect([1414], 600_000)
table(rl, "\nreplay on the lead's sample (seed 1414, 600k)")
P("\nD1 on the lead's sample (g >= 1, k = 0):")
for d in range(1, 9):
    m = (rl["k"] == 0) & (rl["g"] >= 1) & (rl["D"] == d)
    b, r2c, _ = ols([rl["gap"][m], rl["M0"][m]], rl["y"][m])
    P(f"  Delta={d}: n = {m.sum():6d}  c1 = {b[0]:6.1f}  c2 = {b[1]:5.2f}  R2cen = {r2c:.3f}  P(y>0) = {np.mean(rl['y'][m] > 0):.3f}")
out.close()
