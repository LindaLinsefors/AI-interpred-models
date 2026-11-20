"""E5 (part 2). Which (source, age) cells of the exact decomposition are needed?  Thinning experiments.
Sources: X, P4, P5, P9, clips of the 13 ordinary units (17 source types) x 10 event steps.
seeds 1070..1073, 4 x 32768 = 131072 N(0,1) sequences.  Metrics as in e5_reduced.py."""
from src import *

SRC = ["X", "P4", "P5", "P9"] + [f"c{u}" for u in ORD]
GRP = {"X": [0], "P4": [1], "P5": [2], "P9": [3], "C": [4 + ORD.index(j) for j in COMP], "CW": [4 + ORD.index(i) for i in WAVE]}


def full(x):
    H, Z, p, c = amplitudes(x)
    CX, CP, CC = contributions(x, p, c)
    T = np.concatenate([CX[:, :, None], CP, CC], 2).astype(np.float32)     # (B, 10 steps, 17, 10)
    return T, logits(H)


batches = []
for sd in range(1070, 1074):
    x = sample(32768, sd)
    T, L = full(x)
    batches.append((T, L, second_argmax(x)))
print("additivity check:", max(np.abs(T.sum((1, 2)) - L).max() for T, L, _ in batches))


def score(keep):
    """keep: (10 ages, 17) 0/1 mask.  Returns agreement, task accuracy, centred R2 (0..9), centred R2 (0..8)."""
    ks = keep[AGE].astype(np.float32)                    # index by step
    ag = ta = 0.0; num = den = num8 = den8 = 0.0; n = 0
    for T, L, tgt in batches:
        Lr = np.einsum("bsqp,sq->bp", T, ks).astype(np.float64)
        a = Lr.argmax(1)
        ag += (a == L.argmax(1)).sum(); ta += (a == tgt).sum(); n += len(L)
        c1 = Lr - Lr.mean(1, keepdims=True); c0 = L - L.mean(1, keepdims=True)
        num += ((c1 - c0) ** 2).sum(); den += (c0 ** 2).sum()
        d1 = Lr[:, :9] - Lr[:, :9].mean(1, keepdims=True); d0 = L[:, :9] - L[:, :9].mean(1, keepdims=True)
        num8 += ((d1 - d0) ** 2).sum(); den8 += (d0 ** 2).sum()
    return ag / n, ta / n, 1 - num / den, 1 - num8 / den8


def show(name, keep):
    ag, ta, r2, r28 = score(keep)
    print(f"  {name:58s} agree {ag:.4f}   task acc {ta:.4f}   R2c {r2:7.3f}   R2c(0-8) {r28:7.3f}")


ones = np.ones((10, 17))
show("everything", ones)
print("\n(1) keep only events of age <= A (all sources)")
for Amax in range(10):
    k = ones.copy(); k[Amax + 1:] = 0
    show(f"age <= {Amax}", k)
print("\n(2) drop events of age >= A for ONE source group, keep the rest complete")
for g, idx in GRP.items():
    for Amax in (0, 1, 2, 3, 5, 7):
        k = ones.copy(); k[Amax + 1:, idx] = 0
        show(f"{g}: only age <= {Amax}", k)
print("\n(3) drop events of age <= A for ONE source group (keep its old events)")
for g, idx in GRP.items():
    for Amin in (0, 1, 2):
        k = ones.copy(); k[:Amin + 1, idx] = 0
        show(f"{g}: only age > {Amin}", k)
print("\n(4) wave-unit clips: keep only some units")
for units in ([0], [0, 15], [0, 3, 15], [0, 3, 15, 14], [0, 3, 15, 14, 10, 11], []):
    k = ones.copy(); k[:, GRP["CW"]] = 0
    for u in units:
        k[:, 4 + ORD.index(u)] = 1
    show("CW units kept: " + (",".join(map(str, units)) or "none"), k)
print("\n(5) combined thinning")
k = ones.copy(); k[1:, 0] = 0
show("X only at age 0", k)
k2 = k.copy(); k2[:, 3] = 0
show("X only at age 0, no P9", k2)
k3 = k.copy(); k3[:, GRP["CW"]] = 0; k3[:, 4 + ORD.index(0)] = 1; k3[:, 4 + ORD.index(15)] = 1; k3[:, 4 + ORD.index(3)] = 1
show("X only at age 0, CW only units 0,3,15", k3)
