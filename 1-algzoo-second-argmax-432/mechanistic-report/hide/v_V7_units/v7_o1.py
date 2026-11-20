"""V7 / claim O1 (independent code).

States (b, t) with M_t > 0.3, a_1 >= 1, a_2 >= 2, binned by g = (M - max(S, 0)) / M.
For every unit, age a_1 = 1..9 and bin: fraction of states with h == 0 exactly, and the median of h / M.
Two poolings are reported: all t pooled, and t = 9 only.
Seeds 5001..5005, 400k sequences each (2M sequences), inputs iid N(0,1).
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

SEEDS = [5001, 5002, 5003, 5004, 5005]
NB = 400_000
BINS = [(0.15, 0.30), (0.45, 0.60), (0.85, 1.0)]
AGES = range(1, 10)


def collect(seed):
    x = sample(NB, seed)
    H = run(x)
    rs = running_stats(x)
    M, S, a1, a2 = rs["M1"], rs["M2"], rs["a1"], rs["a2"]
    sel = (M > 0.3) & (a1 >= 1) & (a2 >= 2)
    g = np.where(sel, (M - np.maximum(S, 0)) / np.where(sel, M, 1.0), np.nan)
    b, t = np.nonzero(sel)
    return dict(r=(H[b, t] / M[b, t, None]).astype(np.float32), zero=(H[b, t] == 0), g=g[b, t],
                a1=a1[b, t], a2=a2[b, t], t=t, seed=np.full(b.size, seed))


parts = [collect(s) for s in SEEDS]
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
print(f"seeds {SEEDS}, {NB} sequences each; selected states: {D['g'].size}")


def tables(mask, label):
    """frac_zero[bin, age, unit], med[bin, age, unit], n[bin, age]."""
    fz = np.full((3, 10, N), np.nan)
    med = np.full((3, 10, N), np.nan)
    n = np.zeros((3, 10), int)
    for k, (lo, hi) in enumerate(BINS):
        mb = mask & (D["g"] > lo) & (D["g"] <= hi)
        for a in AGES:
            m = mb & (D["a1"] == a)
            n[k, a] = m.sum()
            if n[k, a]:
                fz[k, a] = D["zero"][m].mean(0)
                med[k, a] = np.median(D["r"][m], axis=0)
    print(f"\n================ {label} ================")
    print("counts per (bin, age 1..9):")
    for k in range(3):
        print(f"  bin {BINS[k]}: {n[k, 1:]}")
    print("\nfraction exactly zero, the four claimed cells (bins 1, 2, 3):")
    for u, a in [(0, 2), (11, 5), (10, 6), (12, 4)]:
        print(f"  unit {u:2d} at a1 = {a}: " + "  ".join(f"{fz[k, a, u]:.4f}" for k in range(3)))
    print("\nfraction exactly zero, every unit x age (rows = age 1..9), per bin:")
    for k in range(3):
        print(f"  bin {BINS[k]}   units: " + " ".join(f"{u:5d}" for u in range(N)))
        for a in AGES:
            print(f"    a1={a}          " + " ".join(f"{fz[k, a, u]:5.2f}" for u in range(N)))
    print("\nmedian of h_j / M, every unit x age, per bin:")
    for k in range(3):
        print(f"  bin {BINS[k]}   units: " + " ".join(f"{u:6d}" for u in range(N)))
        for a in AGES:
            print(f"    a1={a}          " + " ".join(f"{med[k, a, u]:6.2f}" for u in range(N)))
    print("\nlargest median of h_12 / M over ages 1..9: " +
          "  ".join(f"bin{k+1}: {np.nanmax(med[k, 1:, 12]):.2f} (age {1 + np.nanargmax(med[k, 1:, 12])})" for k in range(3)))
    print("half-range over ages 1..9 of the median of h_j / M  [(max - min) / 2]:")
    for u in [1, 8, 6, 7]:
        hr = [(np.nanmax(med[k, 1:, u]) - np.nanmin(med[k, 1:, u])) / 2 for k in range(3)]
        print(f"  unit {u}: bin1 {hr[0]:.2f}   bin2 {hr[1]:.2f}   bin3 {hr[2]:.2f}   ratio bin3/bin1 {hr[2] / hr[0]:.2f}")
    return fz, med, n


all_t = np.ones(D["g"].size, bool)
fz, med, n = tables(all_t, "ALL t POOLED")
tables(D["t"] == 9, "t = 9 ONLY")

# sampling noise: the claimed cells per seed (all t pooled)
print("\n================ per-seed spread (all t pooled) ================")
for s in SEEDS:
    ms = D["seed"] == s
    out = []
    for u, a in [(0, 2), (11, 5), (10, 6), (12, 4)]:
        for k, (lo, hi) in enumerate(BINS):
            m = ms & (D["g"] > lo) & (D["g"] <= hi) & (D["a1"] == a)
            out.append(f"{D['zero'][m, u].mean():.3f}")
    print(f"  seed {s}: u0@2 {out[0:3]}  u11@5 {out[3:6]}  u10@6 {out[6:9]}  u12@4 {out[9:12]}")
for s in SEEDS:
    ms = D["seed"] == s
    line = f"  seed {s}:"
    for u in [12, 1, 8, 6, 7]:
        for k in (0, 2):
            lo, hi = BINS[k]
            meds = np.array([np.median(D["r"][ms & (D["g"] > lo) & (D["g"] <= hi) & (D["a1"] == a), u]) for a in AGES])
            val = meds.max() if u == 12 else (meds.max() - meds.min()) / 2
            line += f"  u{u} bin{k+1} {val:.2f}"
    print(line)
