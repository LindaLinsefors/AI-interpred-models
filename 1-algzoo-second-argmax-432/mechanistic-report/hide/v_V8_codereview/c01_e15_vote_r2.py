"""Code review of e15_two_score_vectors.py part (ii) and fig_mechanism panel (c) (claims S2, S3).

e15 reports ONE R2 per age of the max: pooled over the ten logit indices (centred per index).
Index 9 (the current input, V[9] ~ 58 x_9) has by far the largest variance, so it can dominate.
Here: R2 per index, pooled over indices 0..7 / 0..8, the FITTED C (e15 prints the weight product
U[:, COMP] @ w[COMP] under the label 'C(idx) (coefficient of x_9, from the a1=5 fit)'), B as a
function of d (claim S3 quotes it without a computation), and the effect of dropping the states
in which x_9 is itself the runner-up (then the regressors S and x_9 coincide).
Same sample as e15 (seed 1515) so that the pooled R2 can be compared digit by digit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *

x = sample(500_000, 1515)
H, Z = run(x, return_pre=True)
st = running_stats(x)
t = 9
V = H[:, t][:, COMP] @ U[:, COMP].T
Mx, Sx, a1, p2 = st["M1"][:, t], st["M2"][:, t], st["a1"][:, t], st["p2"][:, t]
allon = (Z[:, t][:, COMP] > 0).all(1) & (Sx > 0.1)
print("fraction of the selected states (all four on, S > 0.1, a1 >= 1) in which x_9 is the runner-up:",
      round(float(np.mean(p2[allon & (a1 >= 1)] == 9)), 4))


def fit(s):
    X = np.column_stack([(Mx - Sx)[s], Sx[s], x[s, t]])
    Y = V[s]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = ((Y - X @ coef) ** 2).sum(0)
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return coef, res, tot, (Y ** 2).sum(0)


for label, extra in (("as e15", np.ones(len(x), bool)), ("x_9 not the runner-up (p2 < 9)", p2 < 9)):
    print(f"\n=================== selection: {label} ===================")
    print("centred R2:  pooled over all 10 indices (e15's number) | indices 0..8 | indices 0..7 | share of total SS in index 9 | per index 0..9")
    Atab = np.full((10, 10), np.nan); Btab = np.full((10, 10), np.nan); Ctab = np.full((10, 10), np.nan)
    R2idx = np.full((10, 10), np.nan)
    for a in range(1, 10):
        s = allon & (a1 == a) & extra
        coef, res, tot, raw = fit(s)
        Atab[a], Btab[a], Ctab[a] = coef
        R2idx[a] = 1 - res / tot
        print(f"   a1={a}: {1 - res.sum() / tot.sum():.3f} | {1 - res[:9].sum() / tot[:9].sum():.3f} | {1 - res[:8].sum() / tot[:8].sum():.3f} | "
              f"{tot[9] / tot.sum():.3f} | " + " ".join(f"{v:6.3f}" for v in R2idx[a]) + f"   n={s.sum()}")
    print("   range over a1 = 2..9:  pooled all 10: %.3f-%.3f | indices 0..8: %.3f-%.3f | indices 0..7: %.3f-%.3f | single index (0..7): %.3f-%.3f, median %.3f" % (
        min(1 - fit(allon & (a1 == a) & extra)[1].sum() / fit(allon & (a1 == a) & extra)[2].sum() for a in range(2, 10)),
        max(1 - fit(allon & (a1 == a) & extra)[1].sum() / fit(allon & (a1 == a) & extra)[2].sum() for a in range(2, 10)),
        min(1 - fit(allon & (a1 == a) & extra)[1][:9].sum() / fit(allon & (a1 == a) & extra)[2][:9].sum() for a in range(2, 10)),
        max(1 - fit(allon & (a1 == a) & extra)[1][:9].sum() / fit(allon & (a1 == a) & extra)[2][:9].sum() for a in range(2, 10)),
        min(1 - fit(allon & (a1 == a) & extra)[1][:8].sum() / fit(allon & (a1 == a) & extra)[2][:8].sum() for a in range(2, 10)),
        max(1 - fit(allon & (a1 == a) & extra)[1][:8].sum() / fit(allon & (a1 == a) & extra)[2][:8].sum() for a in range(2, 10)),
        np.nanmin(R2idx[2:, :8]), np.nanmax(R2idx[2:, :8]), np.nanmedian(R2idx[2:, :8])))
    print("\n  per-index R2 at the max's own index (d = 0) and at d = +-4 (where |A| is largest), a1 = 2..9:")
    for a in range(2, 10):
        row = []
        for d in (0, -4, 4, -1, 1):
            i = 9 - a + d
            row.append(f"d={d:+d}: {R2idx[a, i]:.3f}" if 0 <= i <= 7 else f"d={d:+d}:   .  ")
        print(f"   a1={a}: " + "   ".join(row))
    print("\n  FITTED C(a1, idx) (coefficient of x_9); e15 prints instead U[:, COMP] @ w[COMP] =", np.round(U[:, COMP] @ w[COMP], 1))
    for a in range(1, 10):
        print(f"   a1={a}: " + " ".join(f"{v:7.1f}" for v in Ctab[a]))
    print("  fitted A_9 range (a1 = 2..9): [%.1f, %.1f]   B_9 range: [%.1f, %.1f]   C_9 range: [%.1f, %.1f]" % (
        Atab[2:, 9].min(), Atab[2:, 9].max(), Btab[2:, 9].min(), Btab[2:, 9].max(), Ctab[2:, 9].min(), Ctab[2:, 9].max()))
    print("\n  A and B as functions of d = idx - (9 - a1), averaged over a1 = 2..9 with idx <= 7  (B is not tabulated by e15):")
    byA, byB = {}, {}
    for a in range(2, 10):
        for idx in range(8):
            byA.setdefault(idx - (9 - a), []).append(Atab[a, idx]); byB.setdefault(idx - (9 - a), []).append(Btab[a, idx])
    for d in sorted(byA):
        vA, vB = np.array(byA[d]), np.array(byB[d])
        print(f"    d = {d:+d}: A = {vA.mean():7.1f}   B = {vB.mean():6.1f} (min {vB.min():6.1f}, max {vB.max():6.1f})   B - mean_d B ... "
              f"-60cos = {-60 * np.cos(np.radians(42.75 * d)):6.1f}   -8cos = {-8 * np.cos(np.radians(42.75 * d)):5.1f}")
    dd = np.array(sorted(byB)); Bm = np.array([np.mean(byB[d]) for d in dd]); Am = np.array([np.mean(byA[d]) for d in dd])
    Xc = np.column_stack([np.cos(np.radians(42.75 * dd)), np.sin(np.radians(42.75 * dd)), np.ones(len(dd))])
    for nm, y in (("A(d)", Am), ("B(d)", Bm)):
        c, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        r2 = 1 - ((y - Xc @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"    fit {nm} ~ a cos(42.75 d - phi) + c0 over d = -7..7:  a = {np.hypot(c[0], c[1]):.1f}, phi = {np.degrees(np.arctan2(c[1], c[0])):.0f} deg "
              f"(180 = minimum at d = 0), c0 = {c[2]:.1f}, R2 = {r2:.3f}")
