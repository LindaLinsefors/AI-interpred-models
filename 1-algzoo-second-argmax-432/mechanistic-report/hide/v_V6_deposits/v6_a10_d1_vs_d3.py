"""V6: are D1's c1 and D3's k1 the same two-age function?

D1 (record at t):        y = s_t[old max] - s_t[old runner-up] ~ c1 (M0 - S0) + c2 M0      ->  -dy/dS0 = c1
D3 (new runner-up at t): y = s_t[t]       - s_t[old runner-up] ~ k1 (x_t - S0) + k2 M      ->  -dy/dS0 = k1
Index both by (aM, aS): aS = age at t of the displaced old runner-up; aM = age at t of the element
that sits above it (D1: the old max, now runner-up; D3: the max).
c1 from the controlled per-configuration table (mean over t), k1 from the natural D3 events.
"""
from v6_data import *

out = open(HERE / "a10_d1_vs_d3.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


z = np.load(HERE / "a2_configs.npz")
reg = z["reg"]
c1 = np.full((10, 10), np.nan)
for a in range(1, 10):
    for b in range(1, 10):
        if a != b:
            c1[a, b] = np.mean([reg[t, t - a, t - b, 0] for t in range(max(a, b), 10)])
e = dict(np.load(HERE / "d3_events.npz"))
aM3, aS3 = e["t"] - e["pM"], e["t"] - e["pS"]
k1 = np.full((10, 10), np.nan); nk = np.zeros((10, 10), int)
for a in range(1, 10):
    for b in range(1, 10):
        m = (aM3 == a) & (aS3 == b)
        if m.sum() > 300:
            k1[a, b] = np.linalg.lstsq(np.column_stack([e["dx"][m], e["M"][m]]), e["y"][m], rcond=None)[0][0]
            nk[a, b] = m.sum()
ok = ~np.isnan(c1) & ~np.isnan(k1)
P(f"cells with both estimates: {ok.sum()}")
P(f"corr(c1, k1) over cells = {np.corrcoef(c1[ok], k1[ok])[0, 1]:.4f}; median |c1 - k1| = {np.median(np.abs(c1[ok] - k1[ok])):.1f}; "
  f"max |c1 - k1| = {np.abs(c1[ok] - k1[ok]).max():.1f}; mean c1 {c1[ok].mean():.1f}, mean k1 {k1[ok].mean():.1f}")
for nm, m in (("aM <= 4", lambda a, b: a <= 4), ("aM >= 5", lambda a, b: a >= 5), ("|aM - aS| = 1", lambda a, b: abs(a - b) == 1)):
    sel = np.array([[ok[a, b] and m(a, b) for b in range(10)] for a in range(10)])
    P(f"   {nm}: corr = {np.corrcoef(c1[sel], k1[sel])[0, 1]:.4f}, median |diff| = {np.median(np.abs(c1[sel] - k1[sel])):.1f}, max |diff| = {np.abs(c1[sel] - k1[sel]).max():.1f}")
P("rows aM, columns aS = 1..9: 'c1 / k1'")
for a in range(1, 10):
    P(f"  aM={a}: " + " ".join(f"{c1[a, b]:6.1f}/{k1[a, b]:6.1f}" if ok[a, b] else "            -" for b in range(1, 10)))
# does the cosine law describe k1 as well (both ages >= 2)?
for nm, tab in (("c1 (D1)", c1), ("k1 (D3)", k1)):
    for fam, cond in (("older (aS > aM)", lambda a, b: b > a), ("younger (aS < aM)", lambda a, b: b < a)):
        ys, fs = [], []
        for a in range(2, 10):
            for b in range(2, 10):
                if ok[a, b] and cond(a, b):
                    ys.append(tab[a, b]); fs.append(1 - np.cos(OMEGA * abs(a - b)))
        ys, fs = np.array(ys), np.array(fs)
        K = fs @ ys / (fs @ fs)
        P(f"  {nm}, both ages >= 2, runner-up {fam}: K = {K:.1f}, R2 over cells = {1 - ((ys - K * fs) ** 2).sum() / ((ys - ys.mean()) ** 2).sum():.3f} ({len(ys)} cells)")
out.close()
