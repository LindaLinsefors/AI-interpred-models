"""H3 analysis, part 2: the logit kernel of an N event with quiet followers, by source and by steps-to-the-end."""
from hcommon import *
from h3_event import REL, GROUPS, load_events

d = load_events()
n = len(d["t"])
xs, S, M, a, t = d["x"] - d["S"], d["S"], d["M"], d["a"], d["t"]
quiet = d["qmax"] < 0
print(f"N events {n}; with all followers negative: {quiet.sum()}")


def reg(y, s):
    ok = s & np.isfinite(y)
    if ok.sum() < 200: return np.full(3, np.nan), np.nan, ok.sum()
    b, R2 = lsq(y[ok], [xs[ok], S[ok], M[ok]])
    return b, R2, ok.sum()


print("\n=== mean logit change / M by relative position, QUIET followers (rows: age of max) ===")
print("   a  | " + " ".join(f"  d{r:+d}  " for r in REL) + " |   p_max   p_oldS |  n")
for A_ in range(1, 9):
    s = (a == A_) & quiet
    row = " ".join((f"{np.nanmean(d[f'total_r{r}'][s] / M[s]):7.2f}" if np.isfinite(d[f'total_r{r}'][s]).sum() > 100 else "     . ") for r in REL)
    print(f"   {A_}  | {row} | {np.nanmean(d['total_p1'][s] / M[s]):7.2f} {np.nanmean(d['total_p2'][s] / M[s]):7.2f} | {s.sum()}")

print("\n=== split by source, QUIET followers: mean (logit change / M) at own position t | old runner-up | max ===")
print("   a  | " + " | ".join(f"{g:^20s}" for g in GROUPS))
for A_ in range(1, 9):
    s = (a == A_) & quiet
    print(f"   {A_}  | " + " | ".join(f"{np.nanmean(d[f'{g}_r0'][s] / M[s]):6.2f} {np.nanmean(d[f'{g}_p2'][s] / M[s]):6.2f} {np.nanmean(d[f'{g}_p1'][s] / M[s]):6.2f}" for g in GROUPS))

print("\n=== regression of the total logit change on (x_t - S_old, S_old, M), QUIET followers: alpha beta gamma (R2) ===")
for where, lab in (("r0", "own position t"), ("p2", "old runner-up's position"), ("p1", "max's position"), ("r1", "position t+1"), ("r-1", "position t-1"), ("r-2", "position t-2")):
    print(f"  at the {lab}:")
    for A_ in range(1, 9):
        b, R2, k = reg(d[f"total_{where}"], (a == A_) & quiet)
        print(f"    a={A_}: {b[0]:8.2f} {b[1]:8.2f} {b[2]:8.2f}   (R2 {R2:.3f}, n={k})")

print("\n=== own-position bump per unit of x_t (alpha), QUIET followers, by age of max (rows) and steps to the end d = 9 - t (columns 1..7) ===")
print("   a  | " + " ".join(f"   d={k}  " for k in range(1, 8)))
for A_ in range(1, 9):
    row = ""
    for dd in range(1, 8):
        b, R2, k = reg(d["total_r0"], (a == A_) & quiet & (t == 9 - dd))
        row += f" {b[0]:7.1f} " if np.isfinite(b[0]) else "     .   "
    print(f"   {A_}  | {row}")
print("  mean total logit change at own position / M, same layout:")
for A_ in range(1, 9):
    row = ""
    for dd in range(1, 8):
        s = (a == A_) & quiet & (t == 9 - dd)
        row += f" {np.mean(d['total_r0'][s] / M[s]):7.2f} " if s.sum() > 100 else "     .   "
    print(f"   {A_}  | {row}")

print("\n=== source split of alpha (per unit x_t) at own position | old runner-up | max, QUIET followers ===")
print("   a  | " + " | ".join(f"{g:^20s}" for g in GROUPS))
for A_ in range(1, 9):
    row = ""
    for g in GROUPS:
        row += " ".join(f"{reg(d[f'{g}_{wh}'], (a == A_) & quiet)[0][0]:6.2f}" for wh in ("r0", "p2", "p1")) + " | "
    print(f"   {A_}  | {row}")

print("\n=== size of the sources: clip corrections of the comparators at t are (x_t - theta_j); unit-5 pulse; per age (all followers) ===")
print("   a  | mean h5 | h5 per K*M | share of events with >=1 comparator clipped")
for A_ in range(1, 9):
    s = a == A_
    print(f"   {A_}  | {d['h5'][s].mean():6.2f} | {np.mean(d['h5'][s] / (K * M[s])):.3f} | {np.mean(d['ncomp'][s] > 0):.3f}")

# the tie point: where does the final margin (t over old runner-up) cross zero, as a function of r?
print("\n=== P(final logit[t] > logit[old runner-up]) by r = (x_t - S)/(M - S), quiet followers, by age ===")
r = xs / (M - S)
RB = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
m1 = d["L1_t"] - d["L1_p2"]
print("   a  | " + " ".join(f"r<{RB[i+1]:.2f}" for i in range(len(RB) - 1)) + " | all")
for A_ in range(1, 9):
    row = ""
    for i in range(len(RB) - 1):
        s = (a == A_) & quiet & (r >= RB[i]) & (r < RB[i + 1])
        row += f" {np.mean(m1[s] > 0):.3f} " if s.sum() > 50 else "   .   "
    print(f"   {A_}  | {row} | {np.mean(m1[(a == A_) & quiet] > 0):.3f}")
print("  the same for all followers (only events where x_t is still the final runner-up are meaningful; here: all):")
for A_ in range(1, 9):
    row = ""
    for i in range(len(RB) - 1):
        s = (a == A_) & (r >= RB[i]) & (r < RB[i + 1])
        row += f" {np.mean(m1[s] > 0):.3f} " if s.sum() > 50 else "   .   "
    print(f"   {A_}  | {row} | {np.mean(m1[a == A_] > 0):.3f}")
