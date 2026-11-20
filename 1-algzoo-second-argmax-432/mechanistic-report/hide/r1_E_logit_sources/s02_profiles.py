"""S02. Mean position profiles of the event contributions for fixed arrival positions of the max and runner-up.
For each configuration: mean over sequences (seed 1020.., N(0,1)) of the contribution of each (class group, role) to
every logit, minus its mean over positions 0..8 (position 9 shown but not used for centring unless it is p1 or p2)."""
from src import *

np.set_printoptions(precision=1, suppress=True, linewidth=250)
x = np.concatenate([sample(131072, sd) for sd in (1020, 1021)])
out, L = class_role_pos(x)
order = np.argsort(-x, axis=1)
p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
G = {"X+C": [0, 4, 5, 6, 7], "P4": [1], "P5": [2], "P9": [3], "CW": [8], "all": list(range(9))}
for (a, b, sc) in ((3, 6, "RM"), (2, 7, "RM"), (5, 6, "RM"), (3, 6, "MR"), (2, 7, "MR"), (5, 6, "MR")):
    mk = (p2 == a) & (p1 == b) if sc == "RM" else (p1 == a) & (p2 == b)
    print(f"\n=== {sc}: " + (f"runner-up at {a}, max at {b}" if sc == "RM" else f"max at {a}, runner-up at {b}") + f"  (n = {mk.sum()}) ===")
    print("group  role     " + "".join(f"    p{p} " for p in range(10)))
    for g, idx in G.items():
        for r, rn in enumerate(ROLES):
            v = out[mk][:, idx, r].sum(1).mean(0)
            print(f"{g:5s}  {rn:7s} " + "".join(f"{u:7.1f}" for u in v))
        v = out[mk][:, idx].sum((1, 2)).mean(0)
        print(f"{g:5s}  TOTAL   " + "".join(f"{u:7.1f}" for u in v))
