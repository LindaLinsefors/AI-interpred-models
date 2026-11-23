"""d2: does the tape hold something about EACH older input, beyond the top three?   Quiet final states (t = 9).
For each age a = 2..9, on the states in which the input of that age, x_(9-a), is neither the maximum nor the runner-up:
    depth      (S - max(x_(9-a), 0)) / M      how far below the runner-up it is            (R2)
    near tie   S - x_(9-a) < 0.1 M                                                          (ROC AUC)
    is third   x_(9-a) is the third largest                                                 (ROC AUC)
Sources T, R, F as in dec.py, and two references built from the TRUE top-three summary
    V3 = (a_M, a_S, a_T, S/M, T+/S)   (linear decoder: ages one-hot)       V3+T = V3 together with the tape
T against V3 says whether the tape holds more about the entry than the top-three summary would give;
V3+T against V3 says whether the tape adds anything about the entry once the top three are known.
Output: d2_per_age.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
lin, flex, _ = feats(h)
B = len(x); n_train = int(0.75 * B)
M, S, Tp = st["M"], st["S"], np.maximum(st["T3"], 0)
oh = lambda a, lo, hi: (a[:, None] == np.arange(lo, hi + 1)).astype(float)
ratios = np.stack([S / M, Tp / S], 1)
lin["V3"] = np.hstack([oh(st["aM"], 2, 9), oh(st["aS"], 2, 9), oh(st["aT"], 0, 9), ratios])
flex["V3"] = np.hstack([st["aM"][:, None], st["aS"][:, None], st["aT"][:, None], ratios]).astype(float)
lin["V3+T"], flex["V3+T"] = np.hstack([lin["V3"], lin["T"]]), np.hstack([flex["V3"], flex["T"]])
srcs = ("T", "R", "F", "V3", "V3+T")
out(f"quiet final states {B}; train {n_train} (flexible decoder: first 60000 of the selected), test {B - n_train}")

tasks = []
for a in range(2, 10):
    other = (st["aM"] != a) & (st["aS"] != a)
    xa = x[:, 9 - a]
    out(f"age {a}: other input in {other.mean():.3f} of states; among those: third {(st['aT'] == a)[other].mean():.3f}, "
        f"within 0.1 M of S {((S - xa) / M < 0.1)[other].mean():.3f}, sd of depth {((S - np.maximum(xa, 0)) / M)[other].std():.3f}")
    tasks += [(f"age {a}: depth (S - x+)/M", "reg", (S - np.maximum(xa, 0)) / M, other),
              (f"age {a}: within 0.1 M of S", "auc", ((S - xa) / M < 0.1).astype(int), other),
              (f"age {a}: is the third", "auc", (st["aT"] == a).astype(int), other)]
tasks.sort(key=lambda t: t[0].split(": ")[1])
sc, base = table(tasks, lin, flex, n_train, srcs=srcs, n_jobs=16, out=out, n_flex=60_000)
