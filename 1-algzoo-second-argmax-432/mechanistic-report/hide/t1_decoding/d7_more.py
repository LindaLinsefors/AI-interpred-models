"""d7: remaining targets and calibration.  Quiet final states (t = 9).
A. For each age a = 2..9, over ALL quiet final states: the rank of x_(9-a) among the ten inputs (0 = maximum) and the
   value relative to the runner-up, (max(x_(9-a), 0) - S)/M.   R2, sources T, R, F.
B. The age of the third, a_T, with the flexible decoder: accuracy split by how far the third is below the runner-up.
C. Calibration of the 'controls' of d1: how well are the smallest input's age and value predicted from the TRUE summary of
   the top three, V3 = (a_M, a_S, a_T, S/M, T+/S), and does the tape add to that (V3+T)?
Output: d7_more.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
lin, flex, _ = feats(h)
B = len(x); n_train = int(0.75 * B); NF = 60_000
M, S, T3 = st["M"], st["S"], st["T3"]; Tp = np.maximum(T3, 0)
out(f"quiet final states {B}; train {n_train} (flexible decoder: first {NF}), test {B - n_train}")

out("")
out("A. per age, all quiet final states")
tasks = [(f"age {a}: rank of the input (0..9)", "reg", D["rank"][:, 9 - a].astype(float), None) for a in range(2, 10)]
tasks += [(f"age {a}: (x+ - S)/M", "reg", D["relc"][:, 9 - a], None) for a in range(2, 10)]
table(tasks, lin, flex, n_train, srcs=("T", "R", "F"), n_jobs=16, out=out, n_flex=NF)

out("")
out("B. age of the third a_T, flexible decoder, by the distance of the third below the runner-up (S - T)/M; majority class 0.124")
te = slice(n_train, B)


def pred_aT(s):
    return HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=0).fit(flex[s][:NF], st["aT"][:NF]).predict(flex[s][te])


P = dict(zip("TRF", Parallel(n_jobs=3)(delayed(pred_aT)(s) for s in "TRF")))
gap = ((S - T3) / M)[te]; aT = st["aT"][te]
out(f"   {'(S-T)/M':12s} {'n':>6s}" + "".join(f"{s:>8s}" for s in "TRF"))
for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 99)):
    k = (gap >= lo) & (gap < hi)
    out(f"   {lo:4.2f}-{hi:5.2f}  {k.sum():6d}" + "".join(f"{(P[s] == aT)[k].mean():8.3f}" for s in "TRF"))
out("   by the third's age:")
for a in range(10):
    k = aT == a
    out(f"   a_T = {a}      {k.sum():6d}" + "".join(f"{(P[s] == aT)[k].mean():8.3f}" for s in "TRF"))

out("")
out("C. the smallest input, from the true top-three summary V3 and from V3 plus the tape")
oh = lambda a, lo, hi: (a[:, None] == np.arange(lo, hi + 1)).astype(float)
ratios = np.stack([S / M, Tp / S], 1)
lin["V3"] = np.hstack([oh(st["aM"], 2, 9), oh(st["aS"], 2, 9), oh(st["aT"], 0, 9), ratios])
flex["V3"] = np.hstack([st["aM"][:, None], st["aS"][:, None], st["aT"][:, None], ratios]).astype(float)
lin["V3+T"], flex["V3+T"] = np.hstack([lin["V3"], lin["T"]]), np.hstack([flex["V3"], flex["T"]])
lin["V3+F"], flex["V3+F"] = np.hstack([lin["V3"], lin["F"]]), np.hstack([flex["V3"], flex["F"]])
tasks = [("age of smallest (10 cl)", "cls", D["a_min"], None),
         ("min/M (clipped at -5)", "reg", np.maximum(D["min"], -5), None),
         ("age of smallest, given it is >= 2", "cls", D["a_min"], D["a_min"] >= 2)]
table(tasks, lin, flex, n_train, srcs=("T", "F", "V3", "V3+T", "V3+F"), n_jobs=15, out=out, n_flex=NF)
