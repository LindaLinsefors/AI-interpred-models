"""E5 (part 3). Reduced sums with REFITTED kernels.  [FIT]
Keep only some source types; fit, by least squares on a training set, a free kernel K~[source, step, position] for
each kept source (so the kept sources may absorb what is correlated with the dropped ones), and test on fresh data.
train: seeds 1090..1093 (4 x 65536); test: seeds 1094, 1095 (2 x 65536)."""
from src import *

SRC = ["X", "P4", "P5", "P9"] + [f"c{u}" for u in ORD]


def feats(seeds):
    A_, L_, t_ = [], [], []
    for sd in seeds:
        x = sample(65536, sd)
        H, Z, p, c = amplitudes(x)
        A_.append(np.concatenate([x[:, :, None], p, c], 2))     # (B, 10 steps, 17)
        L_.append(logits(H)); t_.append(second_argmax(x))
    return np.concatenate(A_), np.concatenate(L_), np.concatenate(t_)


Atr, Ltr, ttr = feats(range(1090, 1094))
Ate, Lte, tte = feats(range(1094, 1096))
G = {"X": [0], "P4": [1], "P5": [2], "P9": [3], "C": [4 + ORD.index(j) for j in COMP], "CW": [4 + ORD.index(i) for i in WAVE]}


def run_fit(keep_groups, x_age0_only=False, label=None):
    idx = sum((G[g] for g in keep_groups), [])
    def F(A_):
        f = A_[:, :, idx].reshape(len(A_), -1)
        if x_age0_only and "X" in keep_groups:
            k = idx.index(0)
            cols = [s * len(idx) + k for s in range(9)]
            f = np.delete(f, cols, axis=1)
        return f
    Ftr, Fte = F(Atr), F(Ate)
    beta, *_ = np.linalg.lstsq(Ftr, Ltr, rcond=None)
    Lp = Fte @ beta
    a = Lp.argmax(1)
    c1 = Lp - Lp.mean(1, keepdims=True); c0 = Lte - Lte.mean(1, keepdims=True)
    r2 = 1 - ((c1 - c0) ** 2).sum() / (c0 ** 2).sum()
    d1 = Lp[:, :9] - Lp[:, :9].mean(1, keepdims=True); d0 = Lte[:, :9] - Lte[:, :9].mean(1, keepdims=True)
    r28 = 1 - ((d1 - d0) ** 2).sum() / (d0 ** 2).sum()
    print(f"  {label or '+'.join(keep_groups):34s} features {Ftr.shape[1]:4d}   agree {np.mean(a == Lte.argmax(1)):.4f}   task acc {np.mean(a == tte):.4f}   R2c {r2:6.3f}   R2c(0-8) {r28:6.3f}")


print(f"test network accuracy {np.mean(Lte.argmax(1) == tte):.4f}")
for ks in (["X"], ["P5"], ["P4"], ["P4", "P5"], ["X", "P4", "P5"], ["P4", "P5", "P9"], ["X", "P4", "P5", "P9"], ["C"], ["X", "C"], ["X", "P4", "P5", "C"],
           ["P4", "P5", "C"], ["X", "P4", "P5", "CW"], ["X", "P4", "P5", "P9", "CW"], ["X", "P4", "P5", "C", "CW"], ["X", "P4", "P5", "P9", "C"], ["P4", "P5", "P9", "C", "CW"],
           ["X", "P4", "P5", "P9", "C", "CW"]):
    run_fit(ks)
run_fit(["X", "P4", "P5"], x_age0_only=True, label="X(age 0 only)+P4+P5")
run_fit(["X", "P4", "P5", "C"], x_age0_only=True, label="X(age 0 only)+P4+P5+C")

# ---- what do the refitted kernels of the pulses look like in the X+P4+P5+C model?
idx = sum((G[g] for g in ["X", "P4", "P5", "C"]), [])
Ftr = Atr[:, :, idx].reshape(len(Atr), -1)
beta, *_ = np.linalg.lstsq(Ftr, Ltr, rcond=None)                 # (10 steps * 7, 10 positions)
Kfit = beta.reshape(10, len(idx), 10)                            # [step, source, position]
print("\nrefitted kernels in the X+P4+P5+C model, value at the own position (exact kernel in brackets), by age:")
for nm, col, kern, sh in (("P4", 1, KP[0], 0), ("P5", 2, KP[1], 1)):
    row = []
    for s in range(10):
        own = s - sh
        if 0 <= own <= 9:
            row.append(f"a={9 - s}: {Kfit[s, col, own]:6.2f} [{kern[9 - s, own]:5.2f}]")
    print(f"  {nm}: " + "  ".join(row))
print("refitted P4 kernel, contrast own - mean(other positions) by age: " + "  ".join(
    f"a={9 - s}: {Kfit[s, 1, s] - np.delete(Kfit[s, 1], s).mean():6.2f}" for s in range(10)))
print("exact    P4 kernel, contrast own - mean(other positions) by age: " + "  ".join(
    f"a={9 - s}: {KP[0][9 - s, s] - np.delete(KP[0][9 - s], s).mean():6.2f}" for s in range(10)))
print("refitted P5 kernel, contrast own - mean(other positions) by age: " + "  ".join(
    f"a={9 - s}: {Kfit[s, 2, s - 1] - np.delete(Kfit[s, 2], s - 1).mean():6.2f}" for s in range(1, 10)))
print("exact    P5 kernel, contrast own - mean(other positions) by age: " + "  ".join(
    f"a={9 - s}: {KP[1][9 - s, s - 1] - np.delete(KP[1][9 - s], s - 1).mean():6.2f}" for s in range(1, 10)))
