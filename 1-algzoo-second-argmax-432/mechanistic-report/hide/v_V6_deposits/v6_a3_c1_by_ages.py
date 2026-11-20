"""V6 attack on the cosine law: is the lead of the old max over the old runner-up after a record a
function of the SEPARATION, as claimed, or of the two ages?

Uses the per-configuration D1 regressions of a2_configs.npz (record at t = pM; old max = S at pS,
old runner-up = T at pT; D1 conditions S > 0.3, T > 0.1; same base values in every configuration):
    y = s_t[old max] - s_t[old runner-up] ~ c1 (M0 - S0) + c2 M0.
Ages at the record step t:  aM = t - pM0 (= g + 1),  aS = t - pS0.   Delta = |aM - aS|.
"""
from v6_data import *

out = open(HERE / "a3_c1_by_ages.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


z = np.load(HERE / "a2_configs.npz")
reg = z["reg"]            # [pM, pS, pT] -> c1, c2, R2cen, P(y>0), n, resid sd of y/M0
late = [(t, pm, ps) for t in range(2, 10) for pm in range(t) for ps in range(t) if pm != ps]   # (t, pM0, pS0)
c1 = np.array([reg[c][0] for c in late]); c2 = np.array([reg[c][1] for c in late])
r2 = np.array([reg[c][2] for c in late]); py = np.array([reg[c][3] for c in late]); sd = np.array([reg[c][5] for c in late])
t_ = np.array([c[0] for c in late]); aM = np.array([c[0] - c[1] for c in late]); aS = np.array([c[0] - c[2] for c in late])
Dl = np.abs(aM - aS)
g1 = aM >= 2                                                 # the D1 condition g >= 1

P(f"per-configuration D1 regressions: {len(late)} configurations (t, pM0, pS0), n = {int(reg[late[0]][4])} records each")
P("\n[1] pooled by Delta with equal weight per configuration (g >= 1): mean c1, and its spread over configurations")
P("    Delta nconf   mean c1   min    max    sd | mean c2 | mean R2 | mean P(y>0)  min P(y>0) | K(1-cos) with K = 45")
for d in range(1, 9):
    m = g1 & (Dl == d)
    P(f"    {d:3d}  {m.sum():4d}   {c1[m].mean():7.1f} {c1[m].min():6.1f} {c1[m].max():6.1f} {c1[m].std():6.1f} | {c2[m].mean():6.2f} | "
      f"{r2[m].mean():.3f} | {py[m].mean():.3f}  {py[m].min():.3f} | {45 * (1 - np.cos(OMEGA * d)):6.1f}")

P("\n[2] c1 by (aM = age of old max at the record, aS = age of old runner-up at the record), mean over t  [rows aM, columns aS]")
P("       aS: " + " ".join(f"{a:6d}" for a in range(1, 10)))
for a in range(1, 10):
    row = []
    for b in range(1, 10):
        m = (aM == a) & (aS == b)
        row.append(f"{c1[m].mean():6.1f}" if m.any() else "     -")
    P(f"    aM={a}: " + " ".join(row) + ("   <- g = 0, excluded by D1" if a == 1 else ""))
P("    spread over t (max - min of c1 over the configurations sharing the two ages), same layout")
for a in range(1, 10):
    row = []
    for b in range(1, 10):
        m = (aM == a) & (aS == b)
        row.append(f"{np.ptp(c1[m]):6.1f}" if m.sum() > 1 else "     -")
    P(f"    aM={a}: " + " ".join(row))
P("    P(y > 0) by (aM, aS), mean over t")
for a in range(1, 10):
    row = []
    for b in range(1, 10):
        m = (aM == a) & (aS == b)
        row.append(f"{py[m].mean():6.3f}" if m.any() else "     -")
    P(f"    aM={a}: " + " ".join(row))
P("    centred R2 by (aM, aS), mean over t")
for a in range(1, 10):
    row = []
    for b in range(1, 10):
        m = (aM == a) & (aS == b)
        row.append(f"{r2[m].mean():6.3f}" if m.any() else "     -")
    P(f"    aM={a}: " + " ".join(row))

P("\n[3] variance of the per-configuration c1 (g >= 1 only) explained by different descriptions (R2 over configurations, equal weights)")
y = c1[g1]; d = Dl[g1]; am = aM[g1]; as_ = aS[g1]; tt = t_[g1]
sst = ((y - y.mean()) ** 2).sum()


def r2of(pred):
    return 1 - ((y - pred) ** 2).sum() / sst


def group_fit(keys):
    fit = np.zeros(len(y))
    for k in set(keys):
        m = np.array([kk == k for kk in keys])
        fit[m] = y[m].mean()
    return fit


from scipy.optimize import least_squares

res = {}
res["constant (1 par)"] = np.full(len(y), y.mean())
f = 1 - np.cos(OMEGA * d); K = (f @ y) / (f @ f)
res[f"K (1 - cos(42.75 Delta)), K = {K:.1f} (1 par)"] = K * f
fit = least_squares(lambda p: p[0] * (1 - np.cos(np.radians(p[1]) * d)) - y, [45, 42.75])
res[f"K (1 - cos(w Delta)), free w: K = {fit.x[0]:.1f}, w = {fit.x[1]:.2f} deg (2 par)"] = fit.fun + y
fit = least_squares(lambda p: p[0] * (1 - p[2] ** d * np.cos(np.radians(p[1]) * d)) - y, [45, 42.75, 0.95])
res[f"K (1 - r^Delta cos(w Delta)): K = {fit.x[0]:.1f}, w = {fit.x[1]:.2f}, r = {fit.x[2]:.3f} (3 par)"] = fit.fun + y
a = (d @ y) / (d @ d)
res[f"linear a Delta, a = {a:.1f} (1 par)"] = a * d
X = np.column_stack([d, d ** 2]); b = np.linalg.lstsq(X, y, rcond=None)[0]
res[f"parabola a Delta + b Delta^2: {b[0]:.1f}, {b[1]:.2f} (2 par)"] = X @ b
res["free function of Delta (8 par)"] = group_fit(list(d))
res["free function of signed aM - aS (15 par)"] = group_fit(list(am - as_))
res["free function of aS alone (9 par)"] = group_fit(list(as_))
res["free function of aM alone (8 par)"] = group_fit(list(am))
# additive in the two ages: c1 = f(aM) + h(aS)
X = np.zeros((len(y), 18))
X[np.arange(len(y)), am - 1] = 1; X[np.arange(len(y)), 9 + as_ - 1] = 1
res["additive f(aM) + h(aS) (about 16 par)"] = X @ np.linalg.lstsq(X, y, rcond=None)[0]
res["free function of (aM, aS) (64 cells)"] = group_fit(list(zip(am, as_)))
for k, v in res.items():
    P(f"    {r2of(v):6.3f}   {k}")
P("    same, restricted to aS >= 2 (old runner-up not the input right before the record) and aM >= 2:")
m2 = as_ >= 2
y2 = y[m2]; d2 = d[m2]
sst2 = ((y2 - y2.mean()) ** 2).sum()
f2 = 1 - np.cos(OMEGA * d2); K2 = (f2 @ y2) / (f2 @ f2)
P(f"    {1 - ((y2 - K2 * f2) ** 2).sum() / sst2:6.3f}   K (1 - cos(42.75 Delta)), K = {K2:.1f}")
fit = least_squares(lambda p: p[0] * (1 - np.cos(np.radians(p[1]) * d2)) - y2, [45, 42.75])
P(f"    {1 - (fit.fun ** 2).sum() / sst2:6.3f}   free w: K = {fit.x[0]:.1f}, w = {fit.x[1]:.2f} deg")
gf = np.zeros(len(y2))
for k in set(d2):
    gf[d2 == k] = y2[d2 == k].mean()
P(f"    {1 - ((y2 - gf) ** 2).sum() / sst2:6.3f}   free function of Delta")

P("\n[4] the cosine law against the configurations it gets most wrong (g >= 1): c1 observed vs 45 (1 - cos(42.75 Delta))")
pred = 45 * (1 - np.cos(OMEGA * Dl))
idx = np.flatnonzero(g1)
worst = idx[np.argsort(-np.abs(c1[idx] - pred[idx]))][:14]
for i in worst:
    P(f"    t={late[i][0]} pM0={late[i][1]} pS0={late[i][2]} (aM={aM[i]}, aS={aS[i]}, Delta={Dl[i]}): c1 = {c1[i]:6.1f}, law {pred[i]:6.1f}; "
      f"c2 = {c2[i]:6.2f}; R2 = {r2[i]:.3f}; P(y>0) = {py[i]:.3f}")

P("\n[5] same separation, different configurations: all configurations with Delta = 7 and 8, and Delta = 2 for contrast (g >= 1)")
for dd in (8, 7, 6, 2):
    for i in np.flatnonzero((Dl == dd) & g1):
        P(f"    Delta={dd}: t={late[i][0]} pM0={late[i][1]} pS0={late[i][2]} (aM={aM[i]}, aS={aS[i]}): c1 = {c1[i]:6.1f}, c2 = {c2[i]:6.2f}, "
          f"R2 = {r2[i]:.3f}, P(y>0) = {py[i]:.3f}, resid sd of y/M0 = {sd[i]:.2f}")
out.close()
