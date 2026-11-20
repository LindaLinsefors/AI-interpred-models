"""V6 final checks and summary numbers.
[1] sampling noise of the D1 statistics at Delta = 7 (per-seed spread).
[2] D1 restricted to 'old runner-up at least two steps old at the record' (aS >= 2) vs aS = 1 (natural weights, 4M sequences).
[3] regression design: ratio regression y/M0 ~ c1 r + c2 (r = (M0-S0)/M0), intercept, and linearity in r.
[4] before/after: was the old max behind the old runner-up before the record?
[5] D4: share of the 'answers the third' errors carried by g = 0 and by pM = 9 configurations (all D pooled).
[6] the eight configurations with D = 7 or 8.
[7] common free frequency of the old max's footprint (a8_traces.npz).
"""
from v6_data import *
from scipy.optimize import least_squares

out = open(HERE / "a9_summary.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


e = dict(np.load(HERE / "d1_events.npz"))
y = e["yk"][:, 0]
gap, M0, S0 = e["M0"] - e["S0"], e["M0"], e["S0"]
t, pM, pS, seed = e["t"], e["pM"], e["pS"], e["seed"]
D = np.abs(pM - pS)
aM, aS = t - pM, t - pS
fit2 = lambda m: np.linalg.lstsq(np.column_stack([gap[m], M0[m]]), y[m], rcond=None)[0]


def r2c(m):
    b = fit2(m)
    r = y[m] - b[0] * gap[m] - b[1] * M0[m]
    return 1 - (r ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


P("[1] D1 at Delta = 7: per-seed (400k sequences each) values")
m7 = D == 7
ps = np.array([[fit2(m7 & (seed == s))[0], fit2(m7 & (seed == s))[1], r2c(m7 & (seed == s)), np.mean(y[m7 & (seed == s)] > 0),
                (m7 & (seed == s)).sum()] for s in SEEDS])
P("    c1:     " + " ".join(f"{v:6.1f}" for v in ps[:, 0]) + f"   sd {ps[:, 0].std(ddof=1):.2f}")
P("    c2:     " + " ".join(f"{v:6.2f}" for v in ps[:, 1]) + f"   sd {ps[:, 1].std(ddof=1):.2f}")
P("    R2cen:  " + " ".join(f"{v:6.3f}" for v in ps[:, 2]) + f"   sd {ps[:, 2].std(ddof=1):.3f}")
P("    P(y>0): " + " ".join(f"{v:6.3f}" for v in ps[:, 3]) + f"   sd {ps[:, 3].std(ddof=1):.4f}")
P(f"    n per seed about {ps[:, 4].mean():.0f}; the lead's sample has n = 3308, so its s.e. is about sd * sqrt({ps[:, 4].mean():.0f}/3308) = "
  f"c1 {ps[:, 0].std(ddof=1) * np.sqrt(ps[:, 4].mean() / 3308):.2f}, R2 {ps[:, 2].std(ddof=1) * np.sqrt(ps[:, 4].mean() / 3308):.3f}, "
  f"P {ps[:, 3].std(ddof=1) * np.sqrt(ps[:, 4].mean() / 3308):.4f}")
P(f"    pooled over my 10 seeds: c1 {fit2(m7)[0]:.1f}, c2 {fit2(m7)[1]:.2f}, R2 {r2c(m7):.3f}, P(y>0) {np.mean(y[m7] > 0):.3f}  (lead: 41.8, 2.05, 0.36, 0.893)")

P("\n[2] D1 split by the age of the old runner-up at the record (natural weights; s.e. of c1 over the 10 seeds)")
for nm, fam in (("aS >= 2 (old runner-up at least two steps before the record)", aS >= 2), ("aS = 1 (old runner-up is the input right before the record)", aS == 1),
                ("aS >= 2, runner-up OLDER than old max", (aS >= 2) & (aS > aM)), ("aS >= 2, runner-up YOUNGER than old max", (aS >= 2) & (aS < aM))):
    P(f"    {nm}")
    P("      Delta       n     c1  (s.e.)    c2  R2cen  P(y>0)  c1/(1-cos)")
    for d in range(1, 9):
        m = fam & (D == d)
        if m.sum() < 300:
            continue
        b = fit2(m)
        se = np.std([fit2(m & (seed == s))[0] for s in SEEDS], ddof=1) / np.sqrt(len(SEEDS))
        P(f"      {d:5d} {m.sum():7d} {b[0]:6.1f} ({se:4.2f}) {b[1]:6.2f} {r2c(m):6.3f} {np.mean(y[m] > 0):7.3f} {b[0] / (1 - np.cos(OMEGA * d)):10.1f}")
P(f"    share of D1 records with aS = 1: {np.mean(aS == 1):.3f}; by Delta 1..8: " + " ".join(f"{np.mean(aS[D == d] == 1):.2f}" for d in range(1, 9)))

P("\n[3] regression design, by Delta (all D1 records): c1 from (a) the lead's fit, (b) with intercept, (c) ratio regression y/M0 ~ c1 r + c2, (d) ratio regression on r <= 0.3 only")
r = gap / M0
yr = y / M0
for d in range(1, 9):
    m = D == d
    a_ = fit2(m)[0]
    b_ = np.linalg.lstsq(np.column_stack([gap[m], M0[m], np.ones(m.sum())]), y[m], rcond=None)[0]
    c_ = np.polyfit(r[m], yr[m], 1)
    ms = m & (r <= 0.3)
    d_ = np.polyfit(r[ms], yr[ms], 1)
    P(f"    Delta={d}: (a) {a_:6.1f}  (b) {b_[0]:6.1f} [c2 {b_[1]:6.2f}, c0 {b_[2]:5.2f}]  (c) {c_[0]:6.1f} [c2 {c_[1]:6.2f}]  (d) {d_[0]:6.1f} [c2 {d_[1]:6.2f}]")
P("    linearity: median of y/M0 by relative gap r = (M0 - S0)/M0, per Delta  [bins (0,.03] (.03,.08] (.08,.2] (.2,.5] (.5,1)]; then P(y<0) per bin")
edges = [0, .03, .08, .2, .5, 1.0]
for d in range(1, 9):
    med, neg = [], []
    for b in range(5):
        m = (D == d) & (r > edges[b]) & (r <= edges[b + 1])
        med.append(f"{np.median(yr[m]):6.2f}" if m.sum() > 50 else "     -"); neg.append(f"{np.mean(y[m] < 0):.3f}" if m.sum() > 50 else "    -")
    P(f"    Delta={d}: " + " ".join(med) + "   | P(y<0): " + " ".join(neg))

P("\n[4] before the record (t - 1) vs after (t): b = s_{t-1}[old max] - s_{t-1}[old runner-up]  (seed 5030, 400k)")
x = sample(NB, 5030)
H = run(x)
vals, pos = top3(x)
rows = []
for tt in range(2, 10):
    m0, s0 = vals[:, tt - 1, 0], vals[:, tt - 1, 1]
    pm, ps_ = pos[:, tt - 1, 0], pos[:, tt - 1, 1]
    i = np.flatnonzero((x[:, tt] > m0) & (m0 > 0.3) & (s0 > 0.1) & (tt - 1 - pm >= 1))
    Hi = H[i]
    bef = score_at(Hi, tt - 1, pm[i]) - score_at(Hi, tt - 1, ps_[i])
    aft = score_at(Hi, tt, pm[i]) - score_at(Hi, tt, ps_[i])
    rows.append(np.column_stack([np.abs(pm - ps_)[i], bef / m0[i], aft / m0[i], (m0 - s0)[i] / m0[i]]))
rw = np.concatenate(rows)
for d in range(1, 9):
    m = rw[:, 0] == d
    P(f"    Delta={d}: P(before < 0) = {np.mean(rw[m, 1] < 0):.3f}, median before/M0 = {np.median(rw[m, 1]):6.2f}, median after/M0 = {np.median(rw[m, 2]):6.2f}, "
      f"P(after > 0) = {np.mean(rw[m, 2] > 0):.3f}  (n = {m.sum()})")
P(f"    all: P(before < 0) = {np.mean(rw[:, 1] < 0):.3f}, P(after > 0) = {np.mean(rw[:, 2] > 0):.3f}")

z = np.load(HERE / "a2_configs.npz")
cnt, nbin, reg = z["cnt"], z["nbin"], z["reg"]
late = [(pm, ps_, pt) for pm in range(2, 10) for ps_ in range(pm) for pt in range(pm) if ps_ != pt]
P("\n[5] D4 (max last, controlled sampling): share of all 'answers the third' errors carried by sub-families of configurations")
tot = np.array([sum(cnt[c[0], c[1], c[2], b, 1] for c in late) for b in range(4)])
for nm, f in (("g = 0 (pS = pM - 1; outside D1)", lambda c: c[0] - c[1] == 1), ("pM = 9 (record on the last step)", lambda c: c[0] == 9),
              ("g = 0 or pM = 9", lambda c: c[0] - c[1] == 1 or c[0] == 9), ("third right before the max (pT = pM - 1)", lambda c: c[0] - c[2] == 1)):
    sub = [c for c in late if f(c)]
    sh = np.array([sum(cnt[c[0], c[1], c[2], b, 1] for c in sub) for b in range(4)]) / tot
    rate = np.array([np.mean([cnt[c[0], c[1], c[2], b, 1] / nbin[b] for c in sub]) for b in range(4)])
    rest = [c for c in late if not f(c)]
    rate0 = np.array([np.mean([cnt[c[0], c[1], c[2], b, 1] / nbin[b] for c in rest]) for b in range(4)])
    P(f"    {nm:42s}: {len(sub):3d}/{len(late)} configs ({len(sub) / len(late):.0%}); share of errors per bin " + " ".join(f"{v:.2f}" for v in sh)
      + " | error rate inside " + " ".join(f"{v:.3f}" for v in rate) + " | outside " + " ".join(f"{v:.3f}" for v in rate0))
P("    all max-last configurations: error rate per bin " + " ".join(f"{v:.3f}" for v in tot / (len(late) * nbin)))

P("\n[6] the eight configurations with D = 7 or 8 (max last): P(answer = pT) per bin; D1-type regression at the record; cosine-law c1 = 45 (1 - cos(42.75 D))")
for c in ((9, 0, 8), (9, 8, 0), (9, 0, 7), (9, 7, 0), (9, 1, 8), (9, 8, 1), (8, 0, 7), (8, 7, 0)):
    d = abs(c[1] - c[2])
    P(f"    (pS,pT,pM)=({c[1]},{c[2]},{c[0]}) D={d} aM={c[0]-c[1]} aS={c[0]-c[2]}{' [g=0]' if c[0]-c[1]==1 else '      '}: "
      + " ".join(f"{cnt[c[0], c[1], c[2], b, 1] / nbin[b]:.3f}" for b in range(4))
      + f" | c1 = {reg[c][0]:5.1f} (law {45 * (1 - np.cos(OMEGA * d)):4.1f}), c2 = {reg[c][1]:5.2f}, R2 = {reg[c][2]:.3f}, P(y>0) = {reg[c][3]:.3f}")

P("\n[7] old max's mean footprint over slots 0..8 after a record at t = 9 (a8_traces.npz), aM = 2..9: one common frequency for all eight profiles")
zt = np.load(HERE / "a8_traces.npz")
keys, fM = zt["keys"], zt["fM"]
prof = {}
for a in range(2, 10):
    pm = 9 - a
    prof[pm] = np.mean([fM[i][:9] for i in range(len(keys)) if keys[i][0] == pm], 0)
ii = np.arange(9)


def resid(wdeg):
    out_ = []
    for pm, f in prof.items():
        X = np.column_stack([np.cos(np.radians(wdeg) * (ii - pm)), np.sin(np.radians(wdeg) * (ii - pm)), np.ones(9)])
        out_.append(f - X @ np.linalg.lstsq(X, f, rcond=None)[0])
    return np.concatenate(out_)


sst = sum(((f - f.mean()) ** 2).sum() for f in prof.values())
grid = np.arange(25, 70.01, 0.25)
sse = np.array([(resid(wd) ** 2).sum() for wd in grid])
P(f"    best common frequency {grid[sse.argmin()]:.2f} deg per slot (pooled R2 {1 - sse.min() / sst:.3f}); at 42.75: R2 {1 - (resid(42.75) ** 2).sum() / sst:.3f}; "
  f"at 36: {1 - (resid(36.0) ** 2).sum() / sst:.3f}; at 50: {1 - (resid(50.0) ** 2).sum() / sst:.3f}; at 83.28 (2nd harmonic): {1 - (resid(83.28) ** 2).sum() / sst:.3f}")
ok = grid[sse <= sse.min() * 1.1]
P(f"    frequencies whose SSE is within 10% of the minimum: {ok.min():.2f} .. {ok.max():.2f} deg")
out.close()
