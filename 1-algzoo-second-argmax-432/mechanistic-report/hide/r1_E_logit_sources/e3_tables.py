"""E3 tables from e3_cache.npz (run e3_aggregate.py first).  n = 2^19 N(0,1) sequences, seeds 1010..1017."""
from src import *
from e3_aggregate import CLASSES, ROLES, TARGETS, HERE

d = np.load(HERE / "e3_cache.npz")
out, x, L, pos = d["out"].astype(np.float64), d["x"].astype(np.float64), d["L"].astype(np.float64), d["pos"].astype(int)
n = len(x)
p2, p1, p3, q = pos.T
correct = L.argmax(1) == p2
print(f"n = {n}; accuracy = {correct.mean():.4f}")
cls = out.sum(2)                                   # (n, 9 classes, 4 targets)
# margins: runner - max, runner - third, runner - best other
MARG = ["p2-p1", "p2-p3", "p2-best"]
marg = np.stack([cls[:, :, 0] - cls[:, :, 1], cls[:, :, 0] - cls[:, :, 2], cls[:, :, 0] - cls[:, :, 3]], 2)       # (n, 9, 3)
margr = np.stack([out[..., 0] - out[..., 1], out[..., 0] - out[..., 2], out[..., 0] - out[..., 3]], 3)           # (n, 9, 4 roles, 3)
Lsel = np.take_along_axis(L, pos, 1)
print("additivity: max |sum of classes - exact logit| =", np.abs(cls.sum(1) - Lsel).max(), "(float32 cache)")
GROUPS = {"X+C (raw + comparator clips)": [0, 4, 5, 6, 7], "C (4 comparator clips)": [4, 5, 6, 7], "pulses 4+5+9": [1, 2, 3], "everything": list(range(9))}


def table(mask, title):
    m = mask.sum()
    print(f"\n--- {title}  (n = {m}, {m / n:.1%} of all; accuracy {correct[mask].mean():.4f}) ---")
    print("class                         |      L[p2]            L[p1]            L[p3]       |   margin p2-p1     margin p2-p3    margin p2-best  | P(margin p2-best contribution > 0)")
    print("                              |   mean   median    mean   median    mean   median  |   mean   median    mean   median    mean   median |")
    rows = [(c, cls[mask][:, k], marg[mask][:, k]) for k, c in enumerate(CLASSES)]
    rows += [(g, cls[mask][:, idx].sum(1), marg[mask][:, idx].sum(1)) for g, idx in GROUPS.items()]
    for name, a, b in rows:
        print(f"{name:29s} | " + " ".join(f"{a[:, t].mean():7.2f} {np.median(a[:, t]):7.2f} " for t in range(3)) + "| "
              + " ".join(f"{b[:, t].mean():7.2f} {np.median(b[:, t]):7.2f} " for t in range(3)) + f"|  {np.mean(b[:, 2] > 0):.3f}")


def role_table(mask, title, which=(0, 1, 2)):
    print(f"\n--- {title}: MEAN margin contribution split by the role of the causing input (rows: class; columns: role) ---")
    for t in which:
        print(f"  margin {MARG[t]}:   " + "".join(f"{r:>10s}" for r in ROLES) + "     total")
        for k, c in enumerate(CLASSES):
            v = margr[mask][:, k, :, t].mean(0)
            print(f"    {c:4s}            " + "".join(f"{u:10.2f}" for u in v) + f"{v.sum():10.2f}")
        v = margr[mask][:, :, :, t].sum(1).mean(0)
        print(f"    all             " + "".join(f"{u:10.2f}" for u in v) + f"{v.sum():10.2f}")
        v = margr[mask][:, [0, 4, 5, 6, 7]][..., t].sum(1).mean(0)
        print(f"    X+C             " + "".join(f"{u:10.2f}" for u in v) + f"{v.sum():10.2f}")


everything = np.ones(n, bool)
RM = p2 < p1                                         # runner-up arrived before the max (it was the max for a while)
MR = p2 > p1                                         # runner-up arrived after the max
table(everything, "ALL sequences")
table(RM, "scenario RM: runner-up arrived BEFORE the max (p2 < p1)")
table(MR, "scenario MR: runner-up arrived AFTER the max (p2 > p1)")
role_table(everything, "ALL")
role_table(RM, "scenario RM (runner-up before max)")
role_table(MR, "scenario MR (runner-up after max)")

a1, a2 = 9 - p1, 9 - p2
for nm, mk in (("RM, a1 = 0 (max is the last input)", RM & (a1 == 0)), ("RM, a1 = 1", RM & (a1 == 1)), ("RM, a1 >= 2", RM & (a1 >= 2)),
               ("MR, a2 = 0 (runner-up is the last input)", MR & (a2 == 0)), ("MR, a2 = 1", MR & (a2 == 1)), ("MR, a2 >= 2", MR & (a2 >= 2))):
    table(mk, nm)
    role_table(mk, nm, which=(0, 2))

print("\n\n=== mean margin p2-best by class, for each age of the runner-up a2 (rows) ; scenario RM then MR ===")
for nm, sc in (("RM", RM), ("MR", MR)):
    print(f"  {nm}:  a2    n     acc  | " + " ".join(f"{c:>7s}" for c in CLASSES) + " |   total    X+C")
    for a in range(10):
        mk = sc & (a2 == a)
        if mk.sum() < 100:
            continue
        v = marg[mk][:, :, 2].mean(0)
        print(f"       {a:3d} {mk.sum():6d} {correct[mk].mean():.3f} | " + " ".join(f"{u:7.2f}" for u in v) + f" | {v.sum():7.2f} {v[[0, 4, 5, 6, 7]].sum():7.2f}")
print("\n=== mean margin p2-p1 (runner-up minus max) by class, for each age of the max a1 (rows); scenario RM then MR ===")
for nm, sc in (("RM", RM), ("MR", MR)):
    print(f"  {nm}:  a1    n     acc  | " + " ".join(f"{c:>7s}" for c in CLASSES) + " |   total    X+C")
    for a in range(10):
        mk = sc & (a1 == a)
        if mk.sum() < 100:
            continue
        v = marg[mk][:, :, 0].mean(0)
        print(f"       {a:3d} {mk.sum():6d} {correct[mk].mean():.3f} | " + " ".join(f"{u:7.2f}" for u in v) + f" | {v.sum():7.2f} {v[[0, 4, 5, 6, 7]].sum():7.2f}")

print("\n=== wave-unit clips individually: mean contribution to the three margins (all sequences) ===")
wv = d["wave"].astype(np.float64)                     # (n, 9, 4)
for k, u in enumerate(WAVE):
    a = wv[:, k]
    print(f"  wave {u:2d}: p2-p1 {np.mean(a[:, 0] - a[:, 1]):6.2f}   p2-p3 {np.mean(a[:, 0] - a[:, 2]):6.2f}   p2-best {np.mean(a[:, 0] - a[:, 3]):6.2f}   rms(p2-best) {np.sqrt(np.mean((a[:, 0] - a[:, 3]) ** 2)):6.2f}")

print("\n=== size of each class: rms over sequences of its contribution to the margin p2-best, and to the logit L[p2] ===")
for k, c in enumerate(CLASSES):
    print(f"  {c:4s}: rms margin {np.sqrt((marg[:, k, 2] ** 2).mean()):7.2f}   rms L[p2] {np.sqrt((cls[:, k, 0] ** 2).mean()):7.2f}")
for g, idx in GROUPS.items():
    print(f"  {g}: rms margin {np.sqrt((marg[:, idx, 2].sum(1) ** 2).mean()):7.2f}   rms L[p2] {np.sqrt((cls[:, idx, 0].sum(1) ** 2).mean()):7.2f}")
