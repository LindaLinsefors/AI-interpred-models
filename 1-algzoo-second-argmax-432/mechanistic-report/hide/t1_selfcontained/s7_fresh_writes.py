"""S7. Two checks of the reading "the tape piece is wrong at the oldest entries for two steps after a write, and
the transient mode corrects it" (S6 is weights only; this is data, quiet final states).
A. When the tape piece alone does not have the runner-up on top: which entry is on top, by age of that entry, for
   a runner-up of age 2 and for older runner-ups.
B. States without a fresh write: a_M >= 4, a_S >= 4 and the two newest inputs below half the runner-up (so unit 5
   has not fired for them).  P(runner-up on top) for the tape piece alone and with parts added.
Run:  python -B s7_fresh_writes.py > s7_fresh_writes.txt"""
from t1_common import *

d = population(800_000, 701); n = len(d["h"])
pc = pieces(d["h"]); full = d["h"] @ U.T; tape = sum(pc[m] for m in TAPE)
top_t = top_entry(tape, d["valid"]); top_f = top_entry(full, d["valid"])
print(f"{n} quiet final states (800000 sequences, seed 701)")
print("\nA. Tape piece alone, states where its top entry (ages >= 2) is NOT the runner-up: age of the entry on top (share of those states)")
for nm, q in (("runner-up of age 2", d["aS"] == 2), ("runner-up of age 3", d["aS"] == 3), ("runner-up of age >= 4", d["aS"] >= 4)):
    bad = q & (top_t != d["iS"]); age = 9 - top_t[bad]
    print(f"   {nm:22s} n {q.sum():6d}  P(tape fails) {bad.sum() / q.sum():.3f}  |  on top instead, by age 2..9: " + " ".join(f"{np.mean(age == a):.2f}" for a in range(2, 10))
          + f"  | it is the maximum in {np.mean(top_t[bad] == d['iM'][bad]):.2f}, the third in {np.mean(top_t[bad] == d['iT'][bad]):.2f}")
q = d["aS"] == 2
for a in (9, 8):
    print(f"   runner-up of age 2, entry of age {a} (not the maximum): mean of (entry - runner-up's entry)/M:  tape {np.mean(((tape[:, 9 - a] - tape[:, 7]) / d['M'])[q & (d['aM'] != a)]):6.1f}"
          f"   trans {np.mean(((pc['trans'][:, 9 - a] - pc['trans'][:, 7]) / d['M'])[q & (d['aM'] != a)]):6.1f}   full {np.mean(((full[:, 9 - a] - full[:, 7]) / d['M'])[q & (d['aM'] != a)]):6.1f}")

print("\nB. P(runner-up on top among ages >= 2) / agreement with the full score's top entry")
x9, x8, S = d["xr"][:, 9], d["xr"][:, 8], d["S"]
sels = {"all quiet final states": np.ones(n, bool), "a_M >= 3 and a_S >= 3": (d["aM"] >= 3) & (d["aS"] >= 3), "a_M >= 4 and a_S >= 4": (d["aM"] >= 4) & (d["aS"] >= 4),
        "a_M >= 4, a_S >= 4, x_8 and x_9 < 0.5 S": (d["aM"] >= 4) & (d["aS"] >= 4) & (x9 < 0.5 * S) & (x8 < 0.5 * S),
        "a_M >= 4, a_S >= 4, x_7, x_8, x_9 < 0.5 S": (d["aM"] >= 4) & (d["aS"] >= 4) & (x9 < 0.5 * S) & (x8 < 0.5 * S) & (d["xr"][:, 7] < 0.5 * S)}
adds = {"tape alone": [], "+trans": ["trans"], "+trans+echo": ["trans", "echo"], "+clock": ["clock"], "+runner": ["runner"], "+level+runner+clock": ["level", "runner", "clock"],
        "+runner+clock+trans": ["runner", "clock", "trans"], "full": OTHER}
print("   selection                                     n      " + "  ".join(f"{k:>12s}" for k in adds))
for nm, q in sels.items():
    row = f"   {nm:42s} {q.sum():7d}  "
    for k, parts in adds.items():
        A = tape + sum(pc[m] for m in parts) if parts else tape
        row += f"  {np.mean((top_entry(A, d['valid']) == d['iS'])[q]):.3f}/{np.mean((top_entry(A, d['valid']) == top_f)[q]):.3f} "
    print(row)
