"""E5 (part 1). Reduced sums: keep only some source classes (with their measured amplitudes and exact kernels) and
see how well the reduced logits reproduce the network.  seeds 1050..1053 (4 x 65536 = 2^18 N(0,1) sequences).

For each kept set S:  Lred = sum of the exact contributions of the classes in S.
  agree  = P(argmax Lred == argmax L)      task = P(argmax Lred == true runner-up position)
  R2c    = R^2 of the per-sequence-centred logits (Lred - mean_p Lred) against (L - mean_p L), pooled over positions
  R2c08  = same over positions 0..8 only (centred over those 9)
"""
from src import *
from itertools import combinations

outs, Ls, xs = [], [], []
for sd in range(1050, 1054):
    x = sample(65536, sd)
    o, L = class_role_pos(x)
    outs.append(o.astype(np.float32)); Ls.append(L); xs.append(x)
out = np.concatenate(outs); L = np.concatenate(Ls); x = np.concatenate(xs)
n = len(x)
tgt = second_argmax(x)
cls = out.sum(2).astype(np.float64)                       # (n, 9, 10)
print(f"n = {n}; network accuracy {np.mean(L.argmax(1) == tgt):.4f}; additivity max err {np.abs(cls.sum(1) - L).max():.1e} (float32 cache)")


def score(Lred):
    a = Lred.argmax(1)
    c1 = Lred - Lred.mean(1, keepdims=True); c0 = L - L.mean(1, keepdims=True)
    r2 = 1 - ((c1 - c0) ** 2).sum() / (c0 ** 2).sum()
    d1 = Lred[:, :9] - Lred[:, :9].mean(1, keepdims=True); d0 = L[:, :9] - L[:, :9].mean(1, keepdims=True)
    r2b = 1 - ((d1 - d0) ** 2).sum() / (d0 ** 2).sum()
    return np.mean(a == L.argmax(1)), np.mean(a == tgt), r2, r2b


def show(name, Lred):
    ag, ta, r2, r2b = score(Lred)
    print(f"  {name:44s} agree {ag:.4f}   task acc {ta:.4f}   R2c {r2:7.3f}   R2c(pos 0-8) {r2b:7.3f}")


idx = {c: k for k, c in enumerate(CLASSES)}
C4 = ["C1", "C6", "C7", "C8"]
print("\n(a) leave one class out")
for c in CLASSES:
    show("all but " + c, cls.sum(1) - cls[:, idx[c]])
show("all but C1,C6,C7,C8", cls.sum(1) - cls[:, [idx[c] for c in C4]].sum(1))
show("all but X,C1,C6,C7,C8", cls.sum(1) - cls[:, [idx[c] for c in C4 + ['X']]].sum(1))
show("all but P9,CW", cls.sum(1) - cls[:, idx["P9"]] - cls[:, idx["CW"]])
show("all but P4,CW  (no MARK)", cls.sum(1) - cls[:, idx["P4"]] - cls[:, idx["CW"]])
show("all but P4,CW,P9", cls.sum(1) - cls[:, idx["P4"]] - cls[:, idx["CW"]] - cls[:, idx["P9"]])
show("all but P4,P5", cls.sum(1) - cls[:, idx["P4"]] - cls[:, idx["P5"]])
show("all but P4,P5,P9", cls.sum(1) - cls[:, idx["P4"]] - cls[:, idx["P5"]] - cls[:, idx["P9"]])

print("\n(b) keep only k classes: best subsets by agreement (the 4 comparator clips counted as 4 classes)")
for k in (1, 2, 3, 4, 5, 6, 7, 8):
    best = []
    for S in combinations(range(9), k):
        Lred = cls[:, list(S)].sum(1)
        best.append((np.mean(Lred.argmax(1) == L.argmax(1)), S))
    best.sort(reverse=True)
    for ag, S in best[:3]:
        show(f"k={k}: " + "+".join(CLASSES[i] for i in S), cls[:, list(S)].sum(1))

print("\n(c) comparator clips as ONE class 'C' (so classes are X, P4, P5, P9, C, CW)")
G = {"X": [0], "P4": [1], "P5": [2], "P9": [3], "C": [4, 5, 6, 7], "CW": [8]}
names = list(G)
for k in (1, 2, 3, 4, 5):
    best = []
    for S in combinations(names, k):
        Lred = cls[:, sum((G[s] for s in S), [])].sum(1)
        best.append((np.mean(Lred.argmax(1) == L.argmax(1)), S))
    best.sort(reverse=True)
    for ag, S in best[:3]:
        show(f"k={k}: " + "+".join(S), cls[:, sum((G[s] for s in S), [])].sum(1))

print("\n(d) keep only events caused by some inputs (all classes): roles")
rl = out.sum(1).astype(np.float64)                          # (n, 4 roles, 10)
for S in ([0], [1], [0, 1], [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]):
    show("roles " + "+".join(ROLES[r] for r in S), rl[:, S].sum(1))
print("\n(e) keep events of the top-2 / top-3 inputs only, and drop classes")
o64 = out.astype(np.float64)
for S in ([0, 1], [0, 1, 2]):
    for drop in ([], ["P9"], ["CW"], ["P9", "CW"], ["X"], ["P4"], ["P5"]):
        keep = [k for k, c in enumerate(CLASSES) if c not in drop]
        show("roles " + "+".join(ROLES[r] for r in S) + " without " + (",".join(drop) or "-"), o64[:, keep][:, :, S].sum((1, 2)))
