"""E3 (targeted checks of the statements about which sources create the margins).  seeds 1080, 1081 (2 x 131072).

Check 1 (scenario RM): the raw input + comparator clips AT THE STEP OF THE MAX (X+C, role max) put their largest
         contribution, among the positions before the max, at the position of the old max (= final runner-up).
Check 2: the clips of the wave units push down the positions of records; where is the minimum of the total wave-clip
         contribution?
Check 3: own-position 'tape' of the pulses: P4 + P5 contributions of an input to its own position, by role.
"""
from src import *

x = np.concatenate([sample(131072, sd) for sd in (1080, 1081)])
n = len(x); r = np.arange(n)
out, L = class_role_pos(x)
order = np.argsort(-x, axis=1)
p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
RM, MR = p2 < p1, p2 > p1
XC = out[:, [0, 4, 5, 6, 7]].sum(1)            # (n, role, pos)
pos = np.arange(10)[None, :]

print("=== Check 1 (RM, max not at the last step so that the age-0 raw kernel is not involved) ===")
V = XC[:, 0].copy()                            # X+C events at the max's step
before = pos < p1[:, None]
Vb = np.where(before, V, -np.inf)
am = Vb.argmax(1)
for nm, mk in (("RM, p1 <= 8, at least 2 earlier positions", RM & (p1 <= 8) & (p1 >= 2)),):
    print(f"{nm}: n = {mk.sum()}")
    print(f"   argmax over earlier positions of (X+C at the max's step) is the runner-up's position: {np.mean(am[mk] == p2[mk]):.3f}"
          f"   (chance = 1/(number of earlier positions): {np.mean(1 / p1[mk]):.3f});  within +-1 position: {np.mean(np.abs(am[mk] - p2[mk]) <= 1):.3f}")
    vm = V[r, p2][mk]; others = (np.where(before & (pos != p2[:, None]), V, 0).sum(1) / np.maximum(before.sum(1) - 1, 1))[mk]
    print(f"   value at the runner-up's position: mean {vm.mean():.2f}, median {np.median(vm):.2f}; mean over the other earlier positions: {others.mean():.2f};"
          f" at the max's own position: {V[r, p1][mk].mean():.2f}")
print("   by distance d = p1 - p2 (age of the old max when the new max arrives): P(argmax = p2), mean value at p2, mean at other earlier positions, mean at p1")
for dd in range(1, 9):
    mk = RM & (p1 - p2 == dd) & (p1 <= 8) & (p1 >= 2)
    vm = V[r, p2][mk]; others = (np.where(before & (pos != p2[:, None]), V, 0).sum(1) / np.maximum(before.sum(1) - 1, 1))[mk]
    print(f"     d = {dd}: n = {mk.sum():6d}   P = {np.mean(am[mk] == p2[mk]):.3f}   at p2 {vm.mean():7.2f}   others {others.mean():7.2f}   at p1 {V[r, p1][mk].mean():7.2f}")
# same thing for MR as a control: clips at the runner-up's step point at the max
V2 = XC[:, 1].copy(); before2 = pos < p2[:, None]
am2 = np.where(before2, V2, -np.inf).argmax(1)
mk = MR & (p2 <= 8) & (p2 >= 2)
print(f"MR (control): argmax over earlier positions of (X+C at the runner-up's step) is the max's position: {np.mean(am2[mk] == p1[mk]):.3f} (chance {np.mean(1 / p2[mk]):.3f});"
      f" value at the max's position: mean {V2[r, p1][mk].mean():.2f}; at the runner-up's own position: {V2[r, p2][mk].mean():.2f}")
# what the clips of a non-top input do (role third, other): tiny?
print(f"size check: rms over positions/sequences of X+C by role: " + ", ".join(f"{ROLES[k]} {np.sqrt((XC[:, k] ** 2).mean()):.1f}" for k in range(4)))

print("\n=== Check 2: wave-unit clips (class CW, all roles summed) ===")
CW = out[:, 8].sum(1)                           # (n, pos)
amin = CW.argmin(1)
for nm, mk in (("all", np.ones(n, bool)), ("RM", RM), ("MR", MR)):
    print(f"{nm}: argmin of the CW contribution over positions is at the max: {np.mean(amin[mk] == p1[mk]):.3f}, at the runner-up: {np.mean(amin[mk] == p2[mk]):.3f}, at the third: {np.mean(amin[mk] == p3[mk]):.3f}")
    print(f"     mean CW contribution at max {CW[r, p1][mk].mean():6.2f}, runner-up {CW[r, p2][mk].mean():6.2f}, third {CW[r, p3][mk].mean():6.2f}, mean over all positions {CW[mk].mean():6.2f}")
# is the position of the running-record chain what is hit?  positions that were records (prefix maxima) vs not
rec = x >= np.maximum.accumulate(x, axis=1)     # x_s is a prefix maximum
print(f"mean CW contribution at positions that were records when they arrived: {CW[rec].mean():.2f}; at non-record positions: {CW[~rec].mean():.2f}")
a1 = 9 - p1
print("mean CW contribution at the max's position by the age of the max (MR): " + "  ".join(f"a1={a}: {CW[r, p1][MR & (a1 == a)].mean():6.1f}" for a in range(1, 10)))
print("mean CW contribution at the max's position by the age of the max (RM): " + "  ".join(f"a1={a}: {CW[r, p1][RM & (a1 == a)].mean():6.1f}" for a in range(0, 9)))
# dependence on the gap: regress CW at the max's position on M - S (MR, a1 >= 3)
M, S, T3 = x[r, p1], x[r, p2], x[r, p3]
mk = MR & (a1 >= 3)
Xr = np.stack([(M - S)[mk], S[mk], np.ones(mk.sum())], 1); y = (CW[r, p1] - CW[r, p2])[mk]
beta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
print(f"[FIT] MR, a1 >= 3: CW contribution to (L[p1] - L[p2]) ~ {beta[0]:.2f} (M - S) {beta[1]:+.2f} S {beta[2]:+.2f};  R^2 = {1 - ((y - Xr @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")

print("\n=== Check 3: the pulses' own-position tape ===")
P45 = out[:, 1] + out[:, 2]                      # (n, role, pos) P4 + P5
for nm, mk in (("RM", RM), ("MR", MR)):
    print(f"{nm}: P4+P5 caused by the max, read at the max's position: mean {P45[r, 0, p1][mk].mean():6.2f} (P4 {out[r, 1, 0, p1][mk].mean():6.2f}, P5 {out[r, 2, 0, p1][mk].mean():6.2f});"
          f"  caused by the runner-up, read at the runner-up's position: mean {P45[r, 1, p2][mk].mean():6.2f} (P4 {out[r, 1, 1, p2][mk].mean():6.2f}, P5 {out[r, 2, 1, p2][mk].mean():6.2f});"
          f"  third at third: {P45[r, 2, p3][mk].mean():6.2f}")
