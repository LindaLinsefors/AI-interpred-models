"""E3 (the clip bump). When a new max arrives at step p1 and clips all four comparators, the raw input and the comparator
clips of that step contribute  x (KX + KS)(a) - sum_j b_j K_j^unit(a),  b_j = W[j].h_{p1-1},  a = 9 - p1   [EXACT].
How large is the bump it leaves at the old max's position, and what sets its size?   seed 1150/1151, 2 x 131072."""
from src import *

x = np.concatenate([sample(131072, sd) for sd in (1150, 1151)])
n = len(x); r = np.arange(n)
H, Z, p, c = amplitudes(x)
L = logits(H)
order = np.argsort(-x, axis=1); p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
RM = (p2 < p1) & (p1 >= 2) & (p1 <= 8)
CX, CP, CC = contributions(x, p, c)
XC_step = CX + CC[:, :, CI].sum(2)                      # (n, step, pos): raw + comparator clips of each step
V = XC_step[r, p1]                                      # events at the max's step  (n, 10)
# exact identity for fully clipped steps
Hprev = np.concatenate([np.zeros((n, 1, 16)), H[:, :-1]], 1)
b = (Hprev @ W.T)[r, p1][:, COMP]                       # (n, 4): comparator drives at the max's step
allclip = (c[r, p1][:, CI] > 0).all(1)
KS = sum(abs(w[j]) * KC[ORD.index(j)] for j in COMP)
Kunit = np.stack([KC[ORD.index(j)] for j in COMP])      # (4, age, pos)
age = 9 - p1
pred = x[r, p1][:, None] * (KX + KS)[age] - np.einsum("bj,jbp->bp", b, Kunit[:, age])
m = RM & allclip
print(f"RM, max at steps 2..8: n = {RM.sum()}; all four comparators clipped at the max's step in {allclip[RM].mean():.4f}")
print(f"[EXACT] identity X+C = x (KX+KS) - sum_j b_j K_j on those: max abs error {np.abs(pred - V)[m].max():.2e}")
small = x[r, p1][:, None] * (KX + KS)[age]
print(f"   size of the first term x (KX+KS): rms {np.sqrt((small[m] ** 2).mean()):.2f}; of the second: rms {np.sqrt(((pred - small)[m] ** 2).mean()):.2f}")
# bump height at the old max's position, relative to the mean over the other earlier positions
pos = np.arange(10)[None, :]
before = pos < p1[:, None]
oth = np.where(before & (pos != p2[:, None]), V, 0).sum(1) / np.maximum(before.sum(1) - 1, 1)
bump = V[r, p2] - oth
rs = running_stats(x)
Mold = rs["M1"][r, p1 - 1]; Sold = np.maximum(rs["M2"][r, p1 - 1], 0)           # top two just before the max arrived
gap_old = Mold - Sold                                                           # old max minus old runner-up (positive part)
Mnew = x[r, p1]
print(f"\nbump := (X+C of the max's step at the old max's position) - (its mean over the other earlier positions);  mean {bump[m].mean():.2f}, median {np.median(bump[m]):.2f}")
for nm, cols in (("gap_old = M_old - S_old+", [gap_old]), ("M_new - M_old", [Mnew - Mold]), ("M_old", [Mold]), ("gap_old and M_new - M_old", [gap_old, Mnew - Mold]),
                 ("gap_old, M_new - M_old, M_old", [gap_old, Mnew - Mold, Mold])):
    X = np.stack([cc[m] for cc in cols] + [np.ones(m.sum())], 1); y = bump[m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  [FIT] bump ~ {nm:32s}: coefficients {np.round(beta[:-1], 2)}, intercept {beta[-1]:.2f}, R^2 = {1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
print("  by d = p1 - p2 (age of the old max when the new one arrives): slope of bump on gap_old (no intercept), R^2, and mean bump")
for dd in range(1, 9):
    mk = m & (p1 - p2 == dd)
    g = gap_old[mk]; y = bump[mk]; s = (g @ y) / (g @ g)
    print(f"     d = {dd}: n = {mk.sum():6d}  slope {s:6.1f}  R^2 {1 - ((y - s * g) ** 2).sum() / ((y - y.mean()) ** 2).sum():6.3f}   mean bump {y.mean():6.2f}   mean gap_old {g.mean():.3f}")
# the same bump expressed with the old clock coordinate
z = coords(Hprev[r, p1])["clock"]
print(f"  correlation of bump with |z_clock| of the state just before the max arrived: {np.corrcoef(bump[m], np.abs(z[m]))[0, 1]:.3f};  with gap_old: {np.corrcoef(bump[m], gap_old[m])[0, 1]:.3f}")
