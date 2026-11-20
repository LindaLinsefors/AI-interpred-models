"""V6 claims D3 and D4.

D3: new runner-up at t (S_{t-1} < x_t < M_{t-1}, S_{t-1} > 0.2):
    y = s_t[t] - s_t[old runner-up] ~ k1 (x_t - S_{t-1}) + k2 M,  t = 2..9.
D4: full sequences, M > S > T at pM, pS, pT with pM > pS, pM > pT, T > 0.2:
    P(answer = pT) by D = |pS - pT| and (S - T)/S.
Writes d3_d4.txt and d4_seqs.npz (all sequences with T > 0.2, any order, for the attack scripts).
"""
from v6_data import *

out = open(HERE / "d3_d4.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


d3 = {k: [] for k in ("seed", "t", "y", "dx", "M", "S0", "pS", "pM")}
d4 = {k: [] for k in ("seed", "pM", "pS", "pT", "M", "S", "T", "ans", "x4", "p4")}
ntot = 0
for sd, x, H in batches():
    vals, pos = top3(x)
    for t in range(2, 10):
        M0, S0 = vals[:, t - 1, 0], vals[:, t - 1, 1]
        pM, pS = pos[:, t - 1, 0], pos[:, t - 1, 1]
        i = np.flatnonzero((x[:, t] > S0) & (x[:, t] < M0) & (S0 > 0.2))
        Hi = H[i]
        y = score_at(Hi, t, np.full(len(i), t)) - score_at(Hi, t, pS[i])
        for k_, a in zip(("seed", "t", "y", "dx", "M", "S0", "pS", "pM"),
                         (np.full(len(i), sd), np.full(len(i), t), y, x[i, t] - S0[i], M0[i], S0[i], pS[i], pM[i])):
            d3[k_].append(a)
    # D4: keep every sequence with T > 0.2
    order = np.argsort(-x, axis=1)
    i = np.flatnonzero(vals[:, 9, 2] > 0.2)
    ans = logits(H[i]).argmax(1)
    for k_, a in zip(("seed", "pM", "pS", "pT", "M", "S", "T", "ans", "x4", "p4"),
                     (np.full(len(i), sd), pos[i, 9, 0], pos[i, 9, 1], pos[i, 9, 2], vals[i, 9, 0], vals[i, 9, 1],
                      vals[i, 9, 2], ans, np.take_along_axis(x[i], order[i, 3:4], 1)[:, 0], order[i, 3])):
        d4[k_].append(a)
    ntot += len(x)
d3 = {k: np.concatenate(v) for k, v in d3.items()}
d4 = {k: np.concatenate(v) for k, v in d4.items()}
np.savez_compressed(HERE / "d4_seqs.npz", **d4)
np.savez_compressed(HERE / "d3_events.npz", **d3)

P(f"seeds {SEEDS[0]}..{SEEDS[-1]}, {NB} sequences each ({ntot} sequences)")
P("\nD3. y = s_t[t] - s_t[old runner-up] ~ k1 (x_t - S_{t-1}) + k2 M_{t-1}, no intercept")
P("  t       n     k1     k2  R2cen  R2unc  P(y>0) | with intercept: k1    k2    k0  R2cen | s.e.(k1) over seeds")
for t in range(2, 10):
    m = d3["t"] == t
    b, r2c, r2u = ols([d3["dx"][m], d3["M"][m]], d3["y"][m])
    bi, r2ci, _ = ols([d3["dx"][m], d3["M"][m]], d3["y"][m], intercept=True)
    per = np.array([ols([d3["dx"][mm], d3["M"][mm]], d3["y"][mm])[0][0] for s in SEEDS for mm in [m & (d3["seed"] == s)]])
    P(f"{t:3d} {m.sum():7d} {b[0]:6.1f} {b[1]:6.2f} {r2c:6.3f} {r2u:6.3f} {np.mean(d3['y'][m] > 0):7.3f} | "
      f"{bi[0]:6.1f} {bi[1]:6.2f} {bi[2]:6.2f} {r2ci:6.3f} | {per.std(ddof=1) / np.sqrt(len(SEEDS)):.2f}")
P("lead: k1 = 34.5, 27.3, 29.2, 39.3, 50.5, 56.9, 61.8, 60.4 for t = 2..9; R2 0.56-0.72")

P("\nD4. P(answer = pT) by D = |pS - pT| and (S - T)/S; pM > pS, pM > pT, T > 0.2")
edges = [0, .03, .08, .2, .5]
late = (d4["pM"] > d4["pS"]) & (d4["pM"] > d4["pT"])
D = np.abs(d4["pS"] - d4["pT"])
rg = (d4["S"] - d4["T"]) / d4["S"]
errT = d4["ans"] == d4["pT"]
lead = np.array([[.183, .066, .029, .005], [.147, .038, .010, .003], [.189, .029, .003, .000], [.245, .020, .001, .000],
                 [.276, .028, .002, .000], [.338, .137, .023, .002], [.371, .171, .125, .066], [.447, .346, .123, .004]])
P("  D | mine (n) per bin (0,.03] (.03,.08] (.08,.2] (.2,.5]   | lead | max |diff| / binomial s.e. of mine")
mine = np.zeros((8, 4))
for d in range(1, 9):
    row, ns, ses = [], [], []
    for b in range(4):
        m = late & (D == d) & (rg > edges[b]) & (rg <= edges[b + 1])
        p = errT[m].mean()
        mine[d - 1, b] = p
        row.append(p); ns.append(m.sum()); ses.append(np.sqrt(max(p * (1 - p), 1e-9) / m.sum()))
    P(f"  {d} | " + "  ".join(f"{p:.3f} ({n})" for p, n in zip(row, ns)) + "   | "
      + " ".join(f"{v:.3f}" for v in lead[d - 1]) + " | diff " + " ".join(f"{p - v:+.3f}" for p, v in zip(row, lead[d - 1]))
      + " | s.e. " + " ".join(f"{v:.4f}" for v in ses))
P(f"  largest |mine - lead| = {np.abs(mine - lead).max():.4f}")
out.close()
