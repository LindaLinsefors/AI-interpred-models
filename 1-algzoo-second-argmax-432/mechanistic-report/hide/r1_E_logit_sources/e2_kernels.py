"""E2. Kernel tables K(age)[position] for every source type, and locality summaries.  [EXACT: weights only]

Rows are ages a = 0..9 (event at step s = 9 - a), columns are positions 0..9.  'own' marks the position of the input
that caused the event: s for raw inputs, unit 4 and clips; s - 1 for units 5 and 9 (they fire one step after)."""
from src import *

np.set_printoptions(precision=2, suppress=True, linewidth=250)


def table(K, own_shift, title, amax=10):
    print(f"\n{title}")
    print("age own |" + "".join(f"    p{p}  " for p in range(10)) + "|  at own   best other (pos)   mean others")
    for a in range(amax):
        own = 9 - a - own_shift
        row = "".join(f"{v:8.2f}" for v in K[a])
        if 0 <= own <= 9:
            oth = np.delete(K[a], own)
            opos = np.delete(np.arange(10), own)
            ext = f"| {K[a, own]:8.2f}   max {oth.max():7.2f} (p{opos[oth.argmax()]})  min {oth.min():7.2f} (p{opos[oth.argmin()]})   {oth.mean():7.2f}"
            print(f" {a}  p{own}  |{row}{ext}")
        else:
            print(f" {a}  --  |{row}|")


table(KX, 0, "RAW INPUT: logits per unit x_s (K = Uo A^a w[ORD])")
table(KP[0], 0, "PULSE 4: logits per unit h_s[4]")
table(KP[1], 1, "PULSE 5: logits per unit h_s[5]  (own = position s-1)")
table(KP[2], 1, "PULSE 9: logits per unit h_s[9]  (own = position s-1)")
for j in COMP:
    table(abs(w[j]) * KC[ORD.index(j)], 0, f"COMPARATOR {j} CLIP: logits per unit of excess (x_s - theta_{j}) = |w_{j}| Uo A^a e_{j},  |w_{j}| = {abs(w[j]):.2f}")
KS = sum(abs(w[j]) * KC[ORD.index(j)] for j in COMP)
table(KS, 0, "SUM OF THE FOUR COMPARATOR CLIP KERNELS (per unit of a common excess)")
table(KS + KX, 0, "SUM OF FOUR COMPARATOR KERNELS + RAW INPUT KERNEL  (= Uo A^a [w restricted to the wave units])")
for i in WAVE:
    table(KC[ORD.index(i)], 0, f"WAVE UNIT {i} CLIP: logits per unit clip amount (Uo A^a e_{i})")

# ---------------------------------------------------------------------------------------------------------
print("\n\n=== locality summary: kernel relative to the own position, K(a)[own + d] ===")
print("for each source: value at own position per age; and 'contrast' = K[own] - mean over the nine other positions")


def rel(K, own_shift):
    """K[a, own + d] for d = -3..3 (nan outside 0..9)."""
    out = np.full((10, 7), np.nan)
    for a in range(10):
        own = 9 - a - own_shift
        for k, d in enumerate(range(-3, 4)):
            if 0 <= own <= 9 and 0 <= own + d <= 9:
                out[a, k] = K[a, own + d]
    return out


srcs = [("raw x", KX, 0), ("pulse 4", KP[0], 0), ("pulse 5", KP[1], 1), ("pulse 9", KP[2], 1)]
srcs += [(f"comp {j} (per excess)", abs(w[j]) * KC[ORD.index(j)], 0) for j in COMP]
srcs += [("sum 4 comps (per excess)", KS, 0)]
srcs += [(f"wave {i} clip", KC[ORD.index(i)], 0) for i in WAVE]
for name, K, sh in srcs:
    R = rel(K, sh)
    print(f"\n{name}: rows = age, columns = offset d = -3..+3 from own position (d<0: earlier positions)")
    for a in range(10):
        own = 9 - a - sh
        if not (0 <= own <= 9):
            continue
        others = np.delete(K[a], own)
        z = (K[a, own] - others.mean()) / (others.std() + 1e-12)
        print(f"  age {a} (own p{own}): " + " ".join("    -- " if np.isnan(v) else f"{v:7.2f}" for v in R[a])
              + f"   | own - mean(others) = {K[a, own] - others.mean():7.2f}, sd(others) = {others.std():5.2f}, rank of own among 10 (1 = largest) = {int((K[a] > K[a, own]).sum()) + 1},"
              f" (1 = smallest) = {int((K[a] < K[a, own]).sum()) + 1}")

# ---------------------------------------------------------------------------------------------------------
print("\n\n=== decay with age: L2 norm over positions of the centred kernel (position-mean removed) ===")
print("source                      " + "".join(f"  a={a}  " for a in range(10)))
for name, K, sh in srcs:
    Kc = K - K.mean(1, keepdims=True)
    print(f"{name:26s}  " + "".join(f"{np.linalg.norm(Kc[a]):7.2f}" for a in range(10)))

print("\n=== mode content of each injection direction (|coordinate| in the lead's gauge) ===")
print("direction                 " + "".join(f"{n:>9s}" for n in NAMES))
dirs = [("w[ORD] (raw input)", w[ORD]), ("W[ORD,4]", W[ORD, 4]), ("W[ORD,5]", W[ORD, 5]), ("W[ORD,9]", W[ORD, 9])]
dirs += [(f"|w_{j}| e_{j}", abs(w[j]) * np.eye(13)[ORD.index(j)]) for j in COMP]
dirs += [("sum_j |w_j| e_j", sum(abs(w[j]) * np.eye(13)[ORD.index(j)] for j in COMP))]
dirs += [(f"e_{i}", np.eye(13)[ORD.index(i)]) for i in WAVE]
for name, v in dirs:
    print(f"{name:24s}  " + "".join(f"{abs(LEFT[n] @ v):9.3f}" for n in NAMES))
print("\nreadout strength per mode: norm over positions 0..9 of the centred U[:,ORD] RIGHT (x2 for pairs)")
for n in NAMES:
    R = Uo @ RIGHT[n]
    R = R - R.mean()
    print(f"  {n:7s} {np.linalg.norm(R) * (2 if IS_COMPLEX[n] else 1):8.3f}")
