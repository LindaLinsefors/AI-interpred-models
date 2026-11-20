"""Verifier V1, attack E4: is a tape entry a score of ITS OWN input (causally), compared by value with the runner-up?

T4 is cross-sectional.  Here one input that is neither max nor runner-up is overwritten:
    x_p' = S + v' M   (v' < 0, so the top two of the whole sequence do not change)
and the sequence is re-run.  Read at t = 9:  d_own = (s'[p] - s'[p2]) / M, the max's entry, and the 7 bystander
entries.  If the tape picture is right, d_own follows the T4 curves as a function of v' (dose-response), and
the bystanders move little compared with the own entry.
Seed 5701, 150k sequences (those with S > 0.2 are used), one random non-top-2 position per sequence.
Run: python -B v1_attack_ownvalue.py -> v1_attack_ownvalue.txt
"""
from vlib import *

P = Tee("v1_attack_ownvalue.txt")
x = sample(150_000, 5701)
rng = np.random.default_rng(5702)
p3, val = top3(x, 9)
p1, p2, M, S = p3[:, 0], p3[:, 1], val[:, 0], val[:, 1]
keep = S > 0.2
x, p1, p2, M, S = x[keep], p1[keep], p2[keep], M[keep], S[keep]
n = len(x); r = np.arange(n)
# random non-top-2 position
u = rng.random((n, 10)); u[r, p1] = -1; u[r, p2] = -1
p = u.argmax(1)
age = 9 - p
grid = np.array([-2.5, -1.2, -0.6, -0.3, -0.2, -0.125, -0.075, -0.035, -0.01])
t4 = {"age 1": {-2.5: -18.3, -0.6: -12.8, -0.2: -9.7, -0.075: -4.7},
      "ages>=2": {-2.5: -23.7, -0.6: -17.7, -0.2: -10.8, -0.075: -5.3, -0.01: -1.25}}
base = run(x)[:, 9] @ U.T
d_base = (base - base[r, p2][:, None]) / M[:, None]
Dall = np.empty((len(grid), n, 10))
for j, vv in enumerate(grid):
    xm = x.copy(); xm[r, p] = S + vv * M
    s = run(xm)[:, 9] @ U.T
    Dall[j] = (s - s[r, p2][:, None]) / M[:, None]
d_own = Dall[:, r, p]                       # (grid, n)
d_max = Dall[:, r, p1]
by = np.ones((n, 10), bool); by[r, p] = False; by[r, p1] = False; by[r, p2] = False   # bystanders

P(f"sequences used: {n} (S_9 > 0.2); one overwritten non-top-2 input each; read at t = 9")
P("\n=== dose-response of the OWN entry: median d_own by v' (in brackets: the cross-sectional T4 median claimed for that v) ===")
P("   v' =            " + "".join(f"{g:9.3f}" for g in grid))
for name, m in (("age 0", age == 0), ("age 1", age == 1), ("ages>=2", age >= 2)):
    row = "".join(f"{np.median(d_own[j, m]):9.2f}" for j in range(len(grid)))
    P(f"   {name:8s} n={m.sum():6d} {row}")
    if name in t4:
        P("            claim T4   " + "".join(f"{t4[name].get(float(g), np.nan):9.2f}" for g in grid))
    if name == "age 0":
        P("            61.6 v     " + "".join(f"{61.6*g:9.2f}" for g in grid))
P("   single ages:")
for a in range(0, 10):
    m = age == a
    P(f"   age {a}    n={m.sum():6d} " + "".join(f"{np.median(d_own[j, m]):9.2f}" for j in range(len(grid))))
P("\n   P(d_own > 0) (the overwritten input outscores the runner-up) by v':")
for name, m in (("age 0", age == 0), ("age 1", age == 1), ("ages>=2", age >= 2)):
    P(f"   {name:8s} " + "".join(f"{np.mean(d_own[j, m] > 0):9.4f}" for j in range(len(grid))))
P("   P(network answer at t=9 is still p2) by v':")
for name, m in (("age 0", age == 0), ("age 1", age == 1), ("ages>=2", age >= 2)):
    P(f"   {name:8s} " + "".join(f"{np.mean(Dall[j][m].argmax(1) == p2[m]):9.4f}" for j in range(len(grid))))

P("\n=== history: does it matter when the overwritten input arrived? ages>=2 only; median d_own by v' ===")
late = p > np.maximum(p1, p2); early = p < np.minimum(p1, p2); mid = ~late & ~early
for name, m in (("after both top-2", late & (age >= 2)), ("between them", mid & (age >= 2)), ("before both", early & (age >= 2))):
    P(f"   {name:17s} n={m.sum():6d} " + "".join(f"{np.median(d_own[j, m]):9.2f}" for j in range(len(grid))))

P("\n=== specificity: move x_p from v' = -2.5 to v' = -0.075 ===")
j0, j1 = 0, int(np.where(grid == -0.075)[0][0])
dO = d_own[j1] - d_own[j0]
dM = d_max[j1] - d_max[j0]
dB = Dall[j1] - Dall[j0]                                 # (n, 10)
P("   own entry: change in d;  the max's entry;  bystanders (7 per sequence): median |change|, mean change, 90th pct |change|; share of bystander |change| > half the own change")
for name, m in (("age 0", age == 0), ("age 1", age == 1), ("ages>=2", age >= 2), ("age 2", age == 2), ("age 5", age == 5), ("age 9", age == 9)):
    b = dB[m][by[m]]
    ownrep = np.repeat(np.abs(dO[m]), by[m].sum(1))
    P(f"   {name:8s} own: median {np.median(dO[m]):8.2f} (IQR {np.percentile(dO[m],25):7.2f}..{np.percentile(dO[m],75):7.2f}) | max entry: median {np.median(dM[m]):6.2f}, median|.| {np.median(np.abs(dM[m])):5.2f}"
      f" | bystanders: median|.| {np.median(np.abs(b)):5.2f}, mean {b.mean():6.2f}, p90|.| {np.percentile(np.abs(b),90):6.2f}; share > own/2: {np.mean(np.abs(b) > 0.5*ownrep):.3f}")
P("   bystander change by offset q - p (ages>=2 of p; median change, median |change|):")
m = age >= 2
off = np.arange(10)[None, :] - p[:, None]
for o in range(-5, 6):
    if o == 0:
        continue
    sel = by & (off == o) & m[:, None]
    if sel.sum() > 100:
        P(f"      q-p={o:+d}: n={sel.sum():6d}  median {np.median(dB[sel]):7.2f}  median|.| {np.median(np.abs(dB[sel])):6.2f}")
P("   share of variance: for ages>=2, var over (sequence, v') of d_own explained by v' alone (between-v' variance of medians / total):")
tot = d_own[:, m].var(); btw = np.array([d_own[j, m].mean() for j in range(len(grid))]).var()
P(f"      total var {tot:.2f}; between-v' var of means {btw:.2f}; ratio {btw/tot:.3f}")

P("\n=== rank order inside the tape at t = 9 (unmodified sequences, S > 0.2) ===")
sc = base.copy()
order = np.argsort(-sc, axis=1)
p3rd = p3[keep][:, 2]
P("   P(largest score at runner-up) = %.4f ; P(second-largest score at: max %.4f, third-largest input %.4f, other %.4f)"
  % (np.mean(order[:, 0] == p2), np.mean(order[:, 1] == p1), np.mean(order[:, 1] == p3rd), np.mean((order[:, 1] != p1) & (order[:, 1] != p3rd))))
# Spearman between score and value among the 8 non-top-2 inputs
xr = x.copy(); sr = sc.copy()
xr[r, p1] = np.nan; xr[r, p2] = np.nan
msk = ~np.isnan(xr)
xv = xr[msk].reshape(n, 8); sv = sr[msk].reshape(n, 8)
rx = np.argsort(np.argsort(xv, 1), 1).astype(float); rs = np.argsort(np.argsort(sv, 1), 1).astype(float)
rho = 1 - 6 * ((rx - rs) ** 2).sum(1) / (8 * 63)
P("   Spearman(score, value) over the 8 non-top-2 inputs of a sequence: mean %.3f  median %.3f  pct 10: %.3f" % (rho.mean(), np.median(rho), np.percentile(rho, 10)))
agev = np.broadcast_to(9 - np.arange(10), x.shape)[msk].reshape(n, 8).astype(float)
ra = np.argsort(np.argsort(agev, 1), 1).astype(float)
rho_a = 1 - 6 * ((ra - rs) ** 2).sum(1) / (8 * 63)
P("   Spearman(score, age) over the same 8 inputs: mean %.3f  median %.3f" % (rho_a.mean(), np.median(rho_a)))
