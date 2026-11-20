"""exploration: wave-unit OFF probability vs S/M within age-of-max cells."""
from cc import *

seed, x, H, Z = next(batches())
ev = events(x); G = Z > 0
a1, a2, M, S = ev["a1"], ev["a2"], ev["M"], ev["S"]
pos = M > 0
s = np.where(pos, np.maximum(S, 0) / np.where(pos, M, 1), np.nan)
edges = np.linspace(0, 1, 11)
for i, ages in ((0, (2,)), (14, (2,)), (12, (4,)), (11, (5,)), (10, (6, 7, 8)), (15, (2, 3, 4, 5)), (3, (1, 2, 4, 5, 6)), (13, (8, 9))):
    print(f"\nunit {i}: P(off | a1, S+/M bin)   bins of width 0.1 from 0 to 1")
    for a in ages:
        row = ""
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = pos & (a1 == a) & (s >= lo) & (s < hi)
            row += f"{(~G[:, :, i])[m].mean():7.3f}"
        print(f"  a1={a}: {row}   n={(pos & (a1 == a)).sum()}")
# effect of the current class at fixed a1
from a02_events import eff_class
eff = eff_class(x, ev)
print("\nP(off | a1, eff class at t) for units 3, 15, 12, 10, 11")
for i in (3, 15, 12, 10, 11, 14, 13, 0):
    print("unit", i)
    for c in (1, 2):
        print(f"   class {['R*','N*','L*'][c]}: " + "".join(f"{(~G[:, :, i])[pos & (a1 == a) & (eff == c)].mean():7.3f}" for a in range(1, 10)))
