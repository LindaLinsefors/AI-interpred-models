"""S07. When are the wave units clipped, relative to the last record (new running max)?  seed 1140, n = 400k N(0,1).
Rows: wave unit; columns: running age of the max at the step (a = t - position of the max of x_0..x_t); steps t >= 3."""
from src import *

x = sample(400_000, 1140)
H, Z, p, c = amplitudes(x)
rs = running_stats(x)
a1 = rs["a1"][:, 3:]; M1 = rs["M1"][:, 3:]; M2 = np.maximum(rs["M2"][:, 3:], 0)
gap = (M1 - M2) / np.maximum(M1, 1e-9)                       # relative gap (M - S+)/M
print("P(clipped) / mean clip amount (over all states, zeros included), by running age of the max; only states with M1 > 0")
print("unit   " + "".join(f"      a={a}     " for a in range(8)))
for i in WAVE:
    ci = c[:, 3:, ORD.index(i)]
    row = ""
    for a in range(8):
        m = (a1 == a) & (M1 > 0)
        row += f"  {np.mean(ci[m] > 0):.2f} / {ci[m].mean():5.2f} "
    print(f"{i:4d}   {row}")
print("\nsame for unit 0, split by the relative gap (M - S+)/M of the running top two (terciles):")
qs = np.quantile(gap[M1 > 0], [1 / 3, 2 / 3])
ci = c[:, 3:, ORD.index(0)]
for lo, hi, nm in ((-1, qs[0], f"gap < {qs[0]:.2f}"), (qs[0], qs[1], f"{qs[0]:.2f}..{qs[1]:.2f}"), (qs[1], 9, f"gap > {qs[1]:.2f}")):
    row = ""
    for a in range(8):
        m = (a1 == a) & (M1 > 0) & (gap > lo) & (gap <= hi)
        row += f"  {np.mean(ci[m] > 0):.2f} / {ci[m].mean():5.2f} "
    print(f"{nm:14s} {row}")
# kernel of unit 0's clip at the position of the max: clip at step t with max age a -> position t - a = own - a
print("\nunit-0 clip kernel at offset -a from the clip step (= the max's position when the max has running age a), event age 0..5:")
for a in (1, 2, 3, 4, 5):
    print(f"  a = {a}: " + " ".join(f"{KC[ORD.index(0)][ev, 9 - ev - a]:6.2f}" if 9 - ev - a >= 0 else "    --" for ev in range(6)))

# ---- rule test: unit 0 is clipped at step t  <=>  x_(t-2) was a new running max (and positive)
rec = (x >= np.maximum.accumulate(x, axis=1)) & (x > 0)          # x_s is a positive prefix maximum
c0 = c[:, :, ORD.index(0)] > 0
for t0 in (2, 3):
    pred = rec[:, t0 - 2:8 - 0][:, : 10 - t0]                    # rec at t-2 for t = t0..9
    act = c0[:, t0:]
    print(f"\nrule 'unit 0 clipped at t  <=>  x_(t-2) is a positive running max', steps t = {t0}..9: agreement {np.mean(pred == act):.5f};"
          f"  P(clip | record at t-2) = {act[pred].mean():.5f};  P(clip | no record at t-2) = {act[~pred].mean():.5f}  (n = {act.size})")
# amount of the unit-0 clip two steps after a record vs that record's pulses
s = np.arange(2, 8)
m = rec[:, s]
amt = c[:, s + 2, ORD.index(0)][m]
h5n = H[:, s + 1, 5][m]; h9n = H[:, s + 1, 9][m]; h4r = H[:, s, 4][m]
X = np.stack([h5n, h9n, np.ones(m.sum())], 1)
beta, *_ = np.linalg.lstsq(X, amt, rcond=None)
print(f"[FIT] unit-0 clip amount two steps after a record ~ {beta[0]:.3f} h5(next step) {beta[1]:+.3f} h9(next step) {beta[2]:+.3f};  R^2 = {1 - ((amt - X @ beta) ** 2).sum() / ((amt - amt.mean()) ** 2).sum():.3f}"
      f"   (weights W[0,5] = {W[0, 5]:.2f}, W[0,9] = {W[0, 9]:.2f}; n = {m.sum()}; mean amount {amt.mean():.2f}, mean h4 of the record {h4r.mean():.2f})")
X = np.stack([h4r, np.ones(m.sum())], 1); beta, *_ = np.linalg.lstsq(X, amt, rcond=None)
print(f"[FIT] same amount ~ {beta[0]:.3f} h4(record) {beta[1]:+.3f};  R^2 = {1 - ((amt - X @ beta) ** 2).sum() / ((amt - amt.mean()) ** 2).sum():.3f}")
for u, lag in ((12, 4), (11, 5), (10, 6)):
    cu = c[:, :, ORD.index(u)] > 0
    a1f = rs["a1"]; ok = (rs["M1"] > 0)
    t = np.arange(lag + 1, 10)
    print(f"unit {u}: P(clipped | running max age = {lag}) = {cu[:, t][(a1f[:, t] == lag) & ok[:, t]].mean():.4f};  P(clipped | age != {lag}, age >= 1) = {cu[:, t][(a1f[:, t] != lag) & (a1f[:, t] >= 1) & ok[:, t]].mean():.4f}")
t = np.arange(3, 10); a1f = rs["a1"][:, t]; okf = rs["M1"][:, t] > 0; c0t = c0[:, t]
print("unit 0, exact fractions: " + "  ".join(f"P(clip | running max age {a}) = {c0t[(a1f == a) & okf].mean():.4f} (n={int(((a1f == a) & okf).sum())})" for a in range(0, 6)))
prev_rec = rec[:, t - 2]
print(f"unit 0 at running max age 0 (a new max arrives now): P(clip | x_(t-2) was a record) = {c0t[(a1f == 0) & okf & prev_rec].mean():.4f}, P(clip | not) = {c0t[(a1f == 0) & okf & ~prev_rec].mean():.4f}")
print(f"unit 0 at running max age 1: P(clip | x_(t-2) was a record) = {c0t[(a1f == 1) & okf & prev_rec].mean():.4f}, P(clip | not) = {c0t[(a1f == 1) & okf & ~prev_rec].mean():.4f}")
