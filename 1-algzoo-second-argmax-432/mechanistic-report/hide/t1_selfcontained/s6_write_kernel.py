"""S6. Weights only: how does a pulse of unit 5 (the unit that writes a score, one step after its input arrived)
spread over the modes?  A pulse h_t[5] = 1 adds W[o, 5] to the 13 non-pulse units at step t + 1 (if their gates are
open) and A^(k-1) W[o, 5] after k steps of the resting map.  Its effect on the scores, U_o A^(k-1) W[o, 5], is split
exactly into the pieces of the modes.  The written input has age k + 1 at that time, i.e. score index 8 - k.
This is the resting linearisation (all 13 gates open, pulse units silent afterwards), not data.
Run:  python -B s6_write_kernel.py > s6_write_kernel.txt"""
from t1_common import *

A = W[np.ix_(ORD, ORD)]
GROUPS = {"tape (alt+short+h2+h3)": TAPE, "trans": ["trans"], "echo": ["echo"], "level+runner+clock": ["level", "runner", "clock"]}
for u in (5, 4):
    print(f"\n===== pulse of unit {u}: mode coordinates of the kick W[o, {u}] (modulus; gauge: a pulse of unit 4 has coordinate 1 in every mode)")
    print("   " + "  ".join(f"{m} {abs(LEFT[m] @ W[ORD, u]):.2f}" for m in NAMES))
    for k in (1, 2, 3, 5):
        kick = np.linalg.matrix_power(A, k - 1) @ W[ORD, u]
        print(f"  {k} step(s) after the pulse; the input that caused it now has age {k + (1 if u == 5 else 0)} = score index {9 - k - (1 if u == 5 else 0)}")
        print("     index:                     " + " ".join(f"{i:7d}" for i in range(10)))
        print("     all modes (the real effect) " + " ".join(f"{v:7.2f}" for v in Uo @ kick))
        for nm, names in GROUPS.items():
            print(f"     {nm:27s} " + " ".join(f"{v:7.2f}" for v in Uo @ projector(names) @ kick))
print("\nReadout patterns p_m[i] - mean over indices 0..7, for the real modes that are not on the tape (logits per unit coordinate, index 0..7):")
for m in ("level", "runner", "trans", "echo"):
    p = np.real(Uo @ RIGHT[m]); print(f"   {m:7s} " + " ".join(f"{v:7.2f}" for v in p[:8] - p[:8].mean()) + f"    | index 8, 9 (same reference): {p[8] - p[:8].mean():.2f} {p[9] - p[:8].mean():.2f}")
p = Uo @ RIGHT["clock"]
print("   clock: 2|p| " + " ".join(f"{2 * abs(v):7.2f}" for v in p[:8]) + "   phase (deg) " + " ".join(f"{np.degrees(np.angle(v)):5.0f}" for v in p[:8]))
