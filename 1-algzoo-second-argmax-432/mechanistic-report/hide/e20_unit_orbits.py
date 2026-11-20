"""E20. The off-centre circle in raw unit coordinates: medians of h_i / M by age of the max,
for three bins of the relative gap g = (M - max(S, 0)) / M, so the top bin (0.85, 1] also holds
every state with S <= 0.  States are pooled over all steps t.
Selection ('quiet' states): M > 0.3, a1 >= 1 (the max is not the current input) and a2 >= 2 (the
runner-up arrived at least two steps ago), so x_t is below S.  This does NOT ensure that the
runner-up has been the same for >= 2 steps: at a1 = 1 it is the old max, which was demoted only
one step ago."""
from common import *
from modes import RIGHT, coords

x = sample(1_500_000, 2020)
H = run(x)
st = running_stats(x)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Sp = np.maximum(np.where(np.isfinite(S), S, 0), 0)
g = (M - Sp) / np.where(M > 0, M, np.nan)
quiet = (M > 0.3) & (a1 >= 1) & (a2 >= 2)
bins = [(0.15, 0.30), (0.45, 0.60), (0.85, 1.0)]
units = [10, 12, 11, 14, 0, 1, 8, 6, 7, 2, 3, 15, 13]
print("selection: M > 0.3, a1 >= 1, a2 >= 2, states pooled over all t;  gap = (M - max(S, 0)) / M")
print("level part per unit (0.93*10.157*RIGHT_level): ", {u: round(0.93 * 10.157 * RIGHT['level'][ORD.index(u)].real, 1) for u in units})
print("clock amplitude per unit gap (10.157*2|RIGHT_clock|): ", {u: round(10.157 * 2 * abs(RIGHT['clock'][ORD.index(u)]), 1) for u in units})
for (lo, hi) in bins:
    print(f"\n gap bin ({lo}, {hi}]: median h_u / M by age of max (rows = age 1..9)")
    print("  age     n   " + "".join(f"  u{u:<4d}" for u in units))
    for a in range(1, 10):
        s = quiet & (a1 == a) & (g > lo) & (g <= hi)
        if s.sum() < 200:
            continue
        med = np.median(H[s] / M[s][:, None], axis=0)
        print(f"   {a}  {s.sum():6d} " + "".join(f"{med[u]:7.1f}" for u in units))
    print("  fraction of steps with the unit exactly 0, by age:")
    for a in range(1, 10):
        s = quiet & (a1 == a) & (g > lo) & (g <= hi)
        if s.sum() < 200:
            continue
        frac0 = (H[s] == 0).mean(0)
        print(f"   {a}         " + "".join(f"{frac0[u]:7.2f}" for u in units))
