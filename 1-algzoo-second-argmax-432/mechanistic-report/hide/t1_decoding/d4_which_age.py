"""d4: WHICH age does each tape coordinate's phase follow?  Quiet final states (t = 9), numpy only.
lock(m, a) = | mean of (z_m/|z_m|) e^{-i w_m a} |  for candidate ages a (1 = phase is w_m a + const):
    a_S   age of the runner-up                      a_M   age of the maximum
    a_new = min(a_M, a_S)  the more recent of the top two      a_old = max(a_M, a_S)  the earlier of the top two
    a_T   age of the third                          a_prev age of the last overtaken maximum (states where one exists)
    a_k4  age of the last step at which unit 4 was active (> 0.05 M) ; a_k5 the same for unit 5
Also the same split by configuration, the amplitude laws |z_m|/M against the age, and how tightly the complex
number z_m/M itself (not only its phase) is fixed by the ages:  eta2 = 1 - E|z - mean(z | group)|^2 / E|z - mean z|^2.
Output: d4_which_age.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
n = 400_000
xs = sample(n, 11); Hs = run(xs); q = prefix_stats(xs, 9)["quiet"]
x, h, st, D = data()
Hq = Hs[q]; M = st["M"]; B = len(x)
c = tape_coords(h)
z = {"short": c[:, 1] + 1j * c[:, 2], "h2": c[:, 3] + 1j * c[:, 4], "h3": c[:, 5] + 1j * c[:, 6],
     "alt": c[:, 0].astype(complex), "clock*": coords(h)["clock"]}
last = lambda act: np.where(act.any(1), np.argmax(act[:, ::-1], 1), -1)            # age of the last active step
ages = {"a_S": st["aS"], "a_M": st["aM"], "a_new": np.minimum(st["aS"], st["aM"]), "a_old": np.maximum(st["aS"], st["aM"]),
        "a_T": st["aT"], "a_prev": D["a_prev"],
        "a_k4": last(Hq[:, :, 4] > 0.05 * M[:, None]), "a_k5": last(Hq[:, :, 5] > 0.05 * M[:, None])}
sel = {"all": np.ones(B, bool), "S before M": D["S_before_M"], "S after M": ~D["S_before_M"]}
out(f"quiet final states {B}.  P(a_k4 = a_M + 0): {(ages['a_k4'] == st['aM']).mean():.3f};  P(a_k5 = a_M - 1): {(ages['a_k5'] == st['aM'] - 1).mean():.3f};"
    f"  P(a_k5 = a_S - 1): {(ages['a_k5'] == st['aS'] - 1).mean():.3f}")

out("\nA. lock(m, a): phase of the coordinate follows w_m * age.  (alt: w = 180 deg, so lock = |mean sign(zeta) (-1)^a|.)  clock* is not a tape mode.")
for sn, mask in sel.items():
    out(f"  {sn}")
    out("    " + f"{'mode':8s}" + "".join(f"{k:>8s}" for k in ages))
    for nm, zm in z.items():
        w = np.pi if nm == "alt" else np.angle(LAM[nm.strip("*")])
        u = zm / np.abs(zm)
        row = [np.abs((u * np.exp(-1j * w * a))[mask & (a >= 0)].mean()) for a in ages.values()]
        out("    " + f"{nm:8s}" + "".join(f"{v:8.3f}" for v in row))

out("\nB. eta2: share of the variance of the complex coordinate z_m / M explained by its mean within groups of states")
groups = {"a_S": st["aS"], "a_M": st["aM"], "(a_M, a_S)": st["aM"] * 10 + st["aS"],
          "(a_M, a_S, a_T)": st["aM"] * 100 + st["aS"] * 10 + st["aT"],
          "(a_M,a_S), 5 bins S/M": st["aM"] * 100 + st["aS"] * 10 + np.minimum((5 * st["S"] / M).astype(int), 4),
          "(a_M,a_S,a_T), 5 bins S/M, 3 bins T+/S": st["aM"] * 100000 + st["aS"] * 10000 + st["aT"] * 1000
          + np.minimum((5 * st["S"] / M).astype(int), 4) * 10 + np.minimum((3 * np.maximum(st["T3"], 0) / st["S"]).astype(int), 2)}
tr = np.arange(B) < int(0.75 * B)


def eta2(v, g):
    """held-out: group means from train, residual on test (groups unseen in train get the global mean)."""
    ug, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv[tr], minlength=len(ug))
    mean = np.where(cnt > 0, (np.bincount(inv[tr], v[tr].real, len(ug)) + 1j * np.bincount(inv[tr], v[tr].imag, len(ug))) / np.maximum(cnt, 1), v[tr].mean())
    return 1 - (np.abs(v[~tr] - mean[inv[~tr]]) ** 2).mean() / (np.abs(v[~tr] - v[tr].mean()) ** 2).mean(), len(ug)


out("    " + f"{'grouping':42s}{'groups':>7s}" + "".join(f"{k:>8s}" for k in z))
for gn, g in groups.items():
    r = [eta2(zm / M, g) for zm in z.values()]
    out("    " + f"{gn:42s}{r[0][1]:7d}" + "".join(f"{v[0]:8.3f}" for v in r))

out("\nC. the short coordinate as a clock hand for a_S:  z_short / M  rotated back by its own eigenvalue angle, e^{-i w a_S}")
w = np.angle(LAM["short"])
for sn, mask in list(sel.items())[1:]:
    out(f"  {sn}:   a_S, n, mean of Re and Im of (z_short/M) e^(-i w a_S), sd of Re and Im, median |z|/M, and P(phase within 21.5 deg of the mean phase)")
    for a in np.unique(st["aS"][mask]):
        k = mask & (st["aS"] == a); v = (z["short"] / M)[k] * np.exp(-1j * w * a); mu = v.mean()
        hit = np.abs(np.angle(v * np.conj(mu))) < w / 2
        out(f"     {a:2d} {k.sum():7d}   {mu.real:6.2f} {mu.imag:6.2f}   {v.real.std():5.2f} {v.imag.std():5.2f}   {np.median(np.abs(v)):5.2f}   {hit.mean():.3f}")

out("\nD. amplitude of the short coordinate against the values, within one configuration (a_M, a_S): Spearman correlation of |z_short|/M with")
from scipy.stats import spearmanr
Tp = np.maximum(st["T3"], 0)
cand = {"S/M": st["S"] / M, "T+/S": Tp / st["S"], "(S-T+)/M": (st["S"] - Tp) / M, "n within 0.25M": D["n25"]}
for cfg in ((2, 5), (5, 2), (3, 7), (7, 3), (4, 8), (8, 4)):
    k = (st["aM"] == cfg[0]) & (st["aS"] == cfg[1])
    out(f"   a_M = {cfg[0]}, a_S = {cfg[1]} (n = {k.sum()}): " + "  ".join(f"{nm} {spearmanr((np.abs(z['short']) / M)[k], v[k])[0]:+.2f}" for nm, v in cand.items()))
