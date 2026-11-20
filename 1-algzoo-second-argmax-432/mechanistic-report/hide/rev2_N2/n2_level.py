"""N2: per-step growth of the level coordinate vs its median by age (section 4.1 'grows by 1% per step')."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import coords

w4 = w[4]
x = sample(500_000, 9231); H, Z = run(x, return_pre=True); G = Z > 0
st = running_stats(x); M, a1 = st["M1"], st["a1"]
lev = coords(H)["level"] / w4
r = lev[:, 1:] / lev[:, :-1]
newmax_next = x[:, 1:] > M[:, :-1]
q = (a1[:, :-1] >= 2) & (M[:, :-1] > 0.3) & ~newmax_next
pt = G[:, :-1, 5] | G[:, :-1, 9]                      # a pulse unit active at step t (it writes into level(t+1))
print("no new max, max age >= 2: median %.4f mean %.4f" % (np.median(r[q]), r[q].mean()))
print("  units 5 and 9 silent at t (share %.3f): median %.4f mean %.4f" % ((~pt)[q].mean(), np.median(r[q & ~pt]), r[q & ~pt].mean()))
print("  unit 5 or 9 active at t (share %.3f): median %.4f mean %.4f" % (pt[q].mean(), np.median(r[q & pt]), r[q & pt].mean()))
k = M > 0.3
print("median level/(w4 M) by age 2..9:", np.round([np.median((lev / M)[k & (a1 == a)]) for a in range(2, 10)], 3))
# follow fixed cohorts: max at position p stays the max to the end
for p in (0, 2, 4):
    c = (st["p1"][:, 9] == p) & (M[:, 9] > 0.3)
    print(f"cohort: final max at position {p} (n={c.sum()}): median level/(w4 M) at ages 2..{9-p}:", np.round([np.median((lev[c, p + a] / M[c, 9])) for a in range(2, 10 - p)], 3))
