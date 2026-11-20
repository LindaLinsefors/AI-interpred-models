"""Verifier V1: T4 follow-up from the saved table (v1_t4_table.npz, seeds 5201..5205).
(a) the max, ages 2..9, v > 0.2, with the lead's own bin edges (read from e13 AFTER my numbers were produced)
(b) spread (IQR) of d inside the v-bins of T4b / T4c
Run: python -B v1_t4.py --save ; python -B v1_t4_extra.py -> v1_t4_extra.txt   (then delete the .npz)
"""
from vlib import *

if not (HERE / "v1_t4_table.npz").exists():
    raise SystemExit("run  python -B v1_t4.py --save  first (writes the 0.5 GB table v1_t4_table.npz)")
P = Tee("v1_t4_extra.txt")
z = np.load(HERE / "v1_t4_table.npz")
D, V, AGE, KIND = z["d"], z["v"], z["age"], z["kind"]

P("=== (a) max entry, ages 2..9, lead's bins (0.2,0.35] (0.35,0.5] (0.5,0.7] (0.7,0.85] (0.85,1.0]; cells with n > 200 ===")
gb = [0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
lo_, hi_ = np.inf, -np.inf
for a in range(2, 10):
    row = []
    for lo, hi in zip(gb[:-1], gb[1:]):
        q = (KIND == 1) & (AGE == a) & (V > lo) & (V <= hi)
        if q.sum() > 200:
            md = float(np.median(D[q])); row.append(f"{md:8.2f}"); lo_, hi_ = min(lo_, md), max(hi_, md)
        else:
            row.append("       .")
    P(f"   age {a}: " + " ".join(row))
P(f"   range of the cell medians: {lo_:.2f} .. {hi_:.2f}   (claim: between -4.5 and -14.5)")

P("\n=== (b) spread of d inside v-bins: median [q25, q75] ===")
for name, m in (("age 0", (KIND == 0) & (AGE == 0)), ("age 1", (KIND == 0) & (AGE == 1)), ("ages>=2", (KIND == 0) & (AGE >= 2))):
    row = []
    for lo, hi in ((-0.10, -0.05), (-0.25, -0.15), (-0.7, -0.5), (-np.inf, -2)):
        q = m & (V > lo) & (V <= hi)
        d = D[q]
        row.append(f"({lo},{hi}]: {np.median(d):7.2f} [{np.percentile(d,25):7.2f}, {np.percentile(d,75):7.2f}]")
    P(f"   {name:8s} " + "  ".join(row))
P("\n   P(d < 0) for 'other' entries, all v < 0, by age class: " + "  ".join(
    f"{n}: {np.mean(D[m] < 0):.4f}" for n, m in (("age 0", (KIND == 0) & (AGE == 0)), ("age 1", (KIND == 0) & (AGE == 1)), ("ages>=2", (KIND == 0) & (AGE >= 2)))))
P("   P(d < 0) for the max entry by age: " + "  ".join(f"a{a}: {np.mean(D[(KIND == 1) & (AGE == a)] < 0):.4f}" for a in range(10)))
