"""V7 attack 3: is the comparators' oscillation with the age of the max PROPORTIONAL to the gap, does it
shrink to nothing as S -> M, and is that scaling special to the comparators?

  A. the lead's own statistic (half-range over ages 1..9 of the median of h_j/M) in finer gap bins that reach
     g -> 0, for all 16 units, pooled over t and at t = 9 only; straight-line fit against the bin's mean g.
  B. the same with a cleaner amplitude: least squares  median(a) ~ c0 + c1 a + A cos(w a) + B sin(w a), w = 42.75 deg.
  C. per cell (a1, a2): h_j ~ alpha (M - S+) + beta S+ ; if the a1-oscillation were purely proportional to the
     gap, beta would not oscillate with a1.
Seeds 5031..5035, 400k each; selection as in O1: M > 0.3, a1 >= 1, a2 >= 2.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

SEEDS = [5031, 5032, 5033, 5034, 5035]
NB = 400_000
OM = np.radians(42.75)
AGES = np.arange(1, 10)


def collect(seed):
    x = sample(NB, seed)
    H = run(x)
    rs = running_stats(x)
    M, S, a1, a2 = rs["M1"], rs["M2"], rs["a1"], rs["a2"]
    b, t = np.nonzero((M > 0.3) & (a1 >= 1) & (a2 >= 2))
    return dict(h=H[b, t].astype(np.float32), M=M[b, t], S=S[b, t], a1=a1[b, t], a2=a2[b, t], t=t)


parts = [collect(s) for s in SEEDS]
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
del parts
Sp = np.maximum(D["S"], 0)
g = (D["M"] - Sp) / D["M"]
r = D["h"] / D["M"][:, None].astype(np.float32)
print(f"seeds {SEEDS}, {NB} each; selected states {g.size}")

BINS = [(0, .02), (.02, .05), (.05, .15), (.15, .30), (.30, .45), (.45, .60), (.60, .85), (.85, .999999), (.999999, 1.0)]
LAB = ["(0,.02]", "(.02,.05]", "(.05,.15]", "(.15,.30]", "(.30,.45]", "(.45,.60]", "(.60,.85]", "(.85,1)", "g=1"]
SHOW = COMP + [0, 10, 11, 12, 14, 15, 2]
X4 = np.stack([np.ones(9), AGES - 5.0, np.cos(OM * AGES), np.sin(OM * AGES)], 1)


def medians(mask):
    med = np.full((len(BINS), 9, N), np.nan)
    gm = np.zeros(len(BINS)); n = np.zeros(len(BINS), int)
    for k, (lo, hi) in enumerate(BINS):
        mb = mask & (g > lo) & (g <= hi)
        gm[k], n[k] = g[mb].mean(), mb.sum()
        for i, a in enumerate(AGES):
            mm = mb & (D["a1"] == a)
            if mm.sum() >= 200:
                med[k, i] = np.median(r[mm], axis=0)
    return med, gm, n


def analyse(mask, label):
    med, gm, n = medians(mask)
    print(f"\n================ {label} ================")
    hr = (np.nanmax(med, 1) - np.nanmin(med, 1)) / 2                      # (bins, units)
    ok = np.isfinite(med[:, :, 0])                                         # ages present in each bin
    coef = np.stack([np.linalg.lstsq(X4[ok[k]], med[k, ok[k]], rcond=None)[0] for k in range(len(BINS))])   # (bins, 4, units)
    print(f"  ages present: {AGES[ok.all(0)]}")
    amp = np.hypot(coef[:, 2], coef[:, 3])
    for name, tab in (("A. half-range over ages 1..9 of the median of h_j/M", hr),
                      ("B. amplitude of the 42.75-deg component of the median of h_j/M", amp)):
        print(f"\n  {name}")
        print("  gap bin     mean g        n   " + "".join(f"  u{j:<5d}" for j in SHOW))
        for k in range(len(BINS)):
            print(f"  {LAB[k]:10s}  {gm[k]:6.3f} {n[k]:8d}   " + "".join(f"{tab[k, j]:8.2f}" for j in SHOW))
        # straight line through the bins with S > 0 and g > 0.02 (bins 1..7), and through the origin
        use = slice(1, 8)
        A2 = np.stack([np.ones(7), gm[use]], 1)
        print("  fit over bins (.02,.05]..(.85,1):  value = a + b g")
        la = "    a (value extrapolated to g=0): "; lb = "    b:                             "; lr = "    a / (a + b)  [share left at g=0]:"
        l0 = "    measured at (0,.02] / measured at (.85,1): "
        for j in SHOW:
            a, b = np.linalg.lstsq(A2, tab[use, j], rcond=None)[0]
            la += f"{a:8.2f}"; lb += f"{b:8.2f}"; lr += f"{a / (a + b):8.2f}"; l0 += f"{tab[0, j] / tab[7, j]:8.2f}"
        print(la); print(lb); print(lr); print(l0)
        print("    ratio bin (.85,1) / bin (.15,.30]: " + "".join(f"{tab[7, j] / tab[3, j]:8.2f}" for j in SHOW)
              + f"    [ratio of mean g: {gm[7] / gm[3]:.2f}]")
    return med


med_all = analyse(np.ones(g.size, bool), "ALL t POOLED (the lead's pooling)")
analyse(D["t"] == 9, "t = 9 ONLY")
analyse(D["a2"] > D["a1"], "ALL t, runner-up OLDER than the max (the max was a record over it)")
analyse(D["a2"] < D["a1"], "ALL t, runner-up YOUNGER than the max")

# ---------------------------------------------------------------- C
print("\n================ C. per cell (a1, a2): h_j ~ alpha (M - S+) + beta S+   (all t pooled, S > 0.05) ================")
pos = D["S"] > 0.05
al = np.full((10, 10, N), np.nan); be = np.full((10, 10, N), np.nan); r2 = np.full((10, 10, N), np.nan)
for a in range(1, 10):
    for c in range(2, 10):
        mm = pos & (D["a1"] == a) & (D["a2"] == c)
        if mm.sum() < 2000:
            continue
        Xr = np.stack([D["M"][mm] - Sp[mm], Sp[mm]], 1)
        Y = D["h"][mm].astype(np.float64)
        co, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
        al[a, c], be[a, c] = co
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[a, c] = 1 - ((Y - Xr @ co) ** 2).sum(0) / (Y ** 2).sum(0)
for j in COMP + [0, 10, 11, 12]:
    print(f"\n  unit {j}:  alpha (rows a1 = 1..9, columns a2 = 2..9)                      |  beta")
    for a in range(1, 10):
        print("    a1=%d  " % a + " ".join(f"{al[a, c, j]:6.1f}" if np.isfinite(al[a, c, j]) else "    . " for c in range(2, 10))
              + "   |  " + " ".join(f"{be[a, c, j]:6.1f}" if np.isfinite(be[a, c, j]) else "    . " for c in range(2, 10)))
    am, bm = np.nanmean(al[1:, 2:, j], 1), np.nanmean(be[1:, 2:, j], 1)
    asd, bsd = np.nanstd(al[1:, 2:, j], 1), np.nanstd(be[1:, 2:, j], 1)
    ca = np.linalg.lstsq(X4, am, rcond=None)[0]; cb = np.linalg.lstsq(X4, bm, rcond=None)[0]
    print(f"    mean over a2:  alpha(a1) = {np.round(am, 1)}  (sd over a2: {np.round(asd, 1)})")
    print(f"                   beta(a1)  = {np.round(bm, 1)}  (sd over a2: {np.round(bsd, 1)})")
    print(f"    half-range over a1: alpha {np.ptp(am) / 2:.2f}, beta {np.ptp(bm) / 2:.2f} (ratio {np.ptp(bm) / np.ptp(am):.2f});  "
          f"42.75-deg amplitude: alpha {np.hypot(ca[2], ca[3]):.2f}, beta {np.hypot(cb[2], cb[3]):.2f} (ratio {np.hypot(cb[2], cb[3]) / np.hypot(ca[2], ca[3]):.2f});  "
          f"median uncentred R2 {np.nanmedian(r2[1:, 2:, j]):.3f}, min {np.nanmin(r2[1:, 2:, j]):.3f}")
    print(f"    spread of beta over a2 at fixed a1 (the runner-up's own age matters): mean sd {bsd.mean():.2f};  "
          f"spread of alpha over a2: mean sd {asd.mean():.2f}")
