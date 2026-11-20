"""V6 extras.
[1] cosine law per sub-family with free frequency / two amplitudes (per-configuration c1 from a2_configs.npz).
[2] D3: is t the right variable?  k1 by t (the lead's table) vs by the ages of the max and of the old runner-up.
[3] D2 on fixed cohorts under the lead's actual selection (g >= 0) - composition control.
"""
from v6_data import *
from scipy.optimize import least_squares

out = open(HERE / "a7_extras.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


# ---------------------------------------------------------------- [1]
z = np.load(HERE / "a2_configs.npz")
reg = z["reg"]
late = [(t, pm, ps) for t in range(2, 10) for pm in range(t) for ps in range(t) if pm != ps]
c1 = np.array([reg[c][0] for c in late])
aM = np.array([c[0] - c[1] for c in late]); aS = np.array([c[0] - c[2] for c in late])
Dl = np.abs(aM - aS).astype(float)
P("[1] per-configuration c1, sub-families: free-frequency and damped fits")
fam = {"runner-up OLDER (aS > aM >= 2)": (aS > aM) & (aM >= 2), "runner-up YOUNGER (aM > aS >= 2)": (aS < aM) & (aS >= 2),
       "both ages >= 2": (aS >= 2) & (aM >= 2)}
for nm, m in fam.items():
    y, d = c1[m], Dl[m]
    sst = ((y - y.mean()) ** 2).sum()
    f = 1 - np.cos(OMEGA * d); K = f @ y / (f @ f)
    r1 = 1 - ((y - K * f) ** 2).sum() / sst
    fr = least_squares(lambda p: p[0] * (1 - np.cos(np.radians(p[1]) * d)) - y, [50, 42.75])
    fd = least_squares(lambda p: p[0] * (1 - p[1] ** d * np.cos(OMEGA * d)) - y, [50, 0.95])
    a = d @ y / (d @ d)
    X = np.column_stack([d, d ** 2]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    gf = np.zeros(len(y))
    for k in set(d):
        gf[d == k] = y[d == k].mean()
    P(f"  {nm} (n = {m.sum()}):")
    P(f"     K (1 - cos(42.75 D)): K = {K:.1f}, R2 = {r1:.3f}")
    P(f"     free w: K = {fr.x[0]:.1f}, w = {fr.x[1]:.2f} deg, R2 = {1 - (fr.fun ** 2).sum() / sst:.3f}")
    P(f"     damped K (1 - r^D cos(42.75 D)): K = {fd.x[0]:.1f}, r = {fd.x[1]:.3f}, R2 = {1 - (fd.fun ** 2).sum() / sst:.3f}")
    P(f"     linear a D: R2 = {1 - ((y - a * d) ** 2).sum() / sst:.3f};  parabola a D + b D^2: R2 = {1 - ((y - X @ b) ** 2).sum() / sst:.3f};  "
      f"free function of D: R2 = {1 - ((y - gf) ** 2).sum() / sst:.3f}")
m = (aS >= 2) & (aM >= 2)
y, d = c1[m], Dl[m]; up = (aS > aM)[m]
f = 1 - np.cos(OMEGA * d)
X = np.column_stack([f * up, f * ~up]); b = np.linalg.lstsq(X, y, rcond=None)[0]
P(f"  both ages >= 2, two amplitudes (older / younger runner-up): K = {b[0]:.1f} / {b[1]:.1f}, R2 = {1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
P("  mean c1 by Delta, runner-up older | younger (both ages >= 2), and the ratio to (1 - cos(42.75 D)):")
for dd in range(1, 8):
    mu, ml = m & (aS > aM) & (Dl == dd), m & (aS < aM) & (Dl == dd)
    ff = 1 - np.cos(OMEGA * dd)
    P(f"     D={dd}: older {c1[mu].mean():6.1f} (ratio {c1[mu].mean() / ff:5.1f}, {mu.sum()} configs, range {c1[mu].min():.1f}..{c1[mu].max():.1f})"
      f" | younger {c1[ml].mean():6.1f} (ratio {c1[ml].mean() / ff:5.1f}, {ml.sum()} configs, range {c1[ml].min():.1f}..{c1[ml].max():.1f})")

# ---------------------------------------------------------------- [2]
e = dict(np.load(HERE / "d3_events.npz"))
y, dx, M, t = e["y"], e["dx"], e["M"], e["t"]
aS3, aM3 = t - e["pS"], t - e["pM"]                  # ages of the old runner-up and of the max at step t
n = len(y)
sst = ((y - y.mean()) ** 2).sum()


def grouped(keys):
    fit = np.zeros(n)
    _, inv = np.unique(keys, return_inverse=True)
    for k in range(inv.max() + 1):
        mm = inv == k
        b = np.linalg.lstsq(np.column_stack([dx[mm], M[mm]]), y[mm], rcond=None)[0]
        fit[mm] = b[0] * dx[mm] + b[1] * M[mm]
    return fit


P(f"\n[2] D3 ({n} new-runner-up events): pooled centred R2 of y ~ k1(.) (x_t - S0) + k2(.) M")
b = np.linalg.lstsq(np.column_stack([dx, M]), y, rcond=None)[0]
P(f"    {1 - ((y - b[0] * dx - b[1] * M) ** 2).sum() / sst:.3f}   one (k1, k2): k1 = {b[0]:.1f}")
for nm, keys in (("per t (the lead's table, 8 groups)", t), ("per age of the max aM (9 groups)", aM3),
                 ("per age of the old runner-up aS (9 groups)", aS3), ("per (aM, aS) (72 groups)", aM3 * 10 + aS3),
                 ("per (t, aM, aS) (240 groups)", t * 100 + aM3 * 10 + aS3)):
    P(f"    {1 - ((y - grouped(keys)) ** 2).sum() / sst:.3f}   {nm}")
P("    k1 by (aM = age of the max, aS = age of the old runner-up) at the event, pooled over t  [rows aM, columns aS]")
P("       aS: " + " ".join(f"{a:6d}" for a in range(1, 10)))
for a in range(1, 10):
    row = []
    for b_ in range(1, 10):
        mm = (aM3 == a) & (aS3 == b_)
        row.append(f"{np.linalg.lstsq(np.column_stack([dx[mm], M[mm]]), y[mm], rcond=None)[0][0]:6.1f}" if mm.sum() > 300 else "     -")
    P(f"    aM={a}: " + " ".join(row))
P("    k1 by age of the max alone (aM = 1..9): " + " ".join(
    f"{np.linalg.lstsq(np.column_stack([dx[aM3 == a], M[aM3 == a]]), y[aM3 == a], rcond=None)[0][0]:6.1f}" for a in range(1, 10)))
P("    k1 by t for a FIXED age of the max (aM = 2): " + " ".join(
    f"t={tt}: {np.linalg.lstsq(np.column_stack([dx[mm], M[mm]]), y[mm], rcond=None)[0][0]:5.1f}"
    for tt in range(3, 10) for mm in [(aM3 == 2) & (t == tt)]))
P("    k1 by t for a FIXED pair (aM, aS) = (1, 2): " + " ".join(
    f"t={tt}: {np.linalg.lstsq(np.column_stack([dx[mm], M[mm]]), y[mm], rcond=None)[0][0]:5.1f}"
    for tt in range(2, 10) for mm in [(aM3 == 1) & (aS3 == 2) & (t == tt)]))

# ---------------------------------------------------------------- [3]
P("\n[3] D2 under the lead's actual selection (g >= 0, top two unchanged), seeds 5000..5004 x 400k")
rows = {k: [] for k in ("t", "g", "D", "gap", "M0", "last")}
yk = []
for sd in SEEDS[:5]:
    x = sample(NB, sd)
    H = run(x)
    vals, pos = top3(x)
    for tt in range(2, 10):
        M0, S0 = vals[:, tt - 1, 0], vals[:, tt - 1, 1]
        pMm, pSs = pos[:, tt - 1, 0], pos[:, tt - 1, 1]
        i = np.flatnonzero((x[:, tt] > M0) & (M0 > 0.3) & (S0 > 0.1))
        Hi = H[i]
        yy = np.full((len(i), 6), np.nan)
        last = np.zeros(len(i), int)                       # largest k with the top two unchanged up to t + k
        alive = np.ones(len(i), bool)
        for k in range(0, min(6, 10 - tt)):
            if k > 0:
                alive &= x[i, tt + k] < M0[i]
            yy[:, k] = score_at(Hi, tt + k, pMm[i]) - score_at(Hi, tt + k, pSs[i])
            last[alive] = k
        yk.append(yy)
        for k_, a in zip(("t", "g", "D", "gap", "M0", "last"),
                         (np.full(len(i), tt), tt - 1 - pMm[i], np.abs(pMm[i] - pSs[i]), (M0 - S0)[i], M0[i], last)):
            rows[k_].append(a)
r = {k: np.concatenate(v) for k, v in rows.items()}
yk = np.concatenate(yk)
P("    all records valid at k (the lead's D2): c1 = " + ", ".join(
    f"{np.linalg.lstsq(np.column_stack([r['gap'][m], r['M0'][m]]), yk[m, k], rcond=None)[0][0]:.1f}" for k in range(6) for m in [r["last"] >= k])
  + "   (lead: 40.7, 42.6, 35.3, 32.1, 28.6, 23.8)")
P("    mean Delta of the records entering each k: " + ", ".join(f"{r['D'][r['last'] >= k].mean():.2f}" for k in range(6))
  + " | mean t: " + ", ".join(f"{r['t'][r['last'] >= k].mean():.2f}" for k in range(6))
  + " | share g = 0: " + ", ".join(f"{np.mean(r['g'][r['last'] >= k] == 0):.2f}" for k in range(6)))
P("    fixed cohorts (records whose top two stay unchanged up to t + K): c1 at k = 0..K")
for K in range(1, 6):
    m = r["last"] >= K
    P(f"      K={K} (n = {m.sum()}): " + ", ".join(
        f"{np.linalg.lstsq(np.column_stack([r['gap'][m], r['M0'][m]]), yk[m, k], rcond=None)[0][0]:.1f}" for k in range(K + 1)))
P("    fixed cohort K = 5 split by g: ")
for nm, mg in (("g = 0", r["g"] == 0), ("g >= 1", r["g"] >= 1)):
    m = (r["last"] >= 5) & mg
    P(f"      {nm} (n = {m.sum()}): " + ", ".join(
        f"{np.linalg.lstsq(np.column_stack([r['gap'][m], r['M0'][m]]), yk[m, k], rcond=None)[0][0]:.1f}" for k in range(6)))
out.close()
