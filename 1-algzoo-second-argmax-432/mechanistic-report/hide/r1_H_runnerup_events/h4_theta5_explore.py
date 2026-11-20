"""H4 exploration: what else, besides (S, M), sets Theta5 (the threshold of unit 5 on x_t)?"""
from hcommon import *

c5 = W[5, ORD] @ w[ORD]
x = sample(400_000, 1018)
e = Ev(x)
pre0 = e.Hprev @ W.T
# state-only threshold: unit 4 excluded (silent below the max), units 5 and 9 at step t evaluated for x_t = 0
pstate = -(e.Hprev[:, :, PULSE] @ W[np.ix_(ORD, PULSE)].T) @ W[5, ORD] - W[5, 5] * np.maximum(pre0[:, :, 5], 0) - W[5, 9] * np.maximum(pre0[:, :, 9], 0)
Th5 = (-(e.Hprev[:, :, ORD] @ A.T) @ W[5, ORD] + pstate) / c5
st = e.st
B = len(x)
# features aligned with step t (prefix x_0..x_{t-1})
lam = LAM["runner"].real
dsum = np.zeros((B, T)); dpos = np.zeros((B, T))        # decayed sums of all previous inputs / of their positive parts
for t in range(1, T):
    dsum[:, t] = lam * dsum[:, t - 1] + x[:, t - 1]
    dpos[:, t] = lam * dpos[:, t - 1] + np.maximum(x[:, t - 1], 0)
To = np.maximum(np.where(np.isfinite(e.To), e.To, 0), 0)
xprev = np.concatenate([np.zeros((B, 1)), x[:, :-1]], 1)
xprev2 = np.concatenate([np.zeros((B, 2)), x[:, :-2]], 1)
ok = (e.tt >= 3) & (e.tt <= 8) & (e.So > 0)
print("R2 of Theta5 on feature sets (no intercept), steps 3<=t<=8 with S_old>0, by age of the max")
sets = {
    "S,M": lambda s: [e.So[s], e.Mo[s]],
    "S,M,T+": lambda s: [e.So[s], e.Mo[s], To[s]],
    "S,M,T+,dsum": lambda s: [e.So[s], e.Mo[s], To[s], dsum[s]],
    "S,M,T+,dsum,x_{t-1},x_{t-2}": lambda s: [e.So[s], e.Mo[s], To[s], dsum[s], xprev[s], xprev2[s]],
    "S,M,T+,dsum,x_{t-1}+,x_{t-2}+": lambda s: [e.So[s], e.Mo[s], To[s], dsum[s], np.maximum(xprev[s], 0), np.maximum(xprev2[s], 0)],
    "S,M,dsum,dpos": lambda s: [e.So[s], e.Mo[s], dsum[s], dpos[s]],
}
for name, f in sets.items():
    row = ""
    for a in range(1, 9):
        s = ok & (e.a1 == a)
        b, R2 = lsq(Th5[s], f(s))
        row += f" {R2:.3f}"
    print(f"  {name:32s}: {row}")
print("\ncoefficients for 'S,M,T+,dsum,x_{t-1}+,x_{t-2}+' by age:")
for a in range(1, 9):
    s = ok & (e.a1 == a)
    b, R2 = lsq(Th5[s], sets["S,M,T+,dsum,x_{t-1}+,x_{t-2}+"](s))
    print(f"  a={a}: " + " ".join(f"{v:7.3f}" for v in b) + f"   R2={R2:.3f}")

# split by the age of the runner-up: does Theta5/S depend on a2 (recency of the runner-up)?
print("\nmedian Theta5/S by (age of max a1, age of runner-up a2) for steps with (M-S)/M in [0.2, 0.6]:")
gm = (e.Mo - e.So) / e.Mo
print("       a2 = " + " ".join(f"{k:6d}" for k in range(1, 10)))
for a in range(1, 9):
    row = ""
    for a2 in range(1, 10):
        s = ok & (e.a1 == a) & (e.a2 == a2) & (gm > 0.2) & (gm < 0.6)
        row += f" {np.median(Th5[s] / e.So[s]):6.2f}" if s.sum() > 300 else "     . "
    print(f"  a1={a}:  {row}")
print("\nIQR of Theta5/S in the same cells:")
for a in range(1, 9):
    row = ""
    for a2 in range(1, 10):
        s = ok & (e.a1 == a) & (e.a2 == a2) & (gm > 0.2) & (gm < 0.6)
        row += f" {np.percentile(Th5[s] / e.So[s], 75) - np.percentile(Th5[s] / e.So[s], 25):6.2f}" if s.sum() > 300 else "     . "
    print(f"  a1={a}:  {row}")
# within a cell: what explains the spread?  regress on T+/S and x_{t-1}+/S
print("\nwithin cells (a1, a2): R2 of Theta5 ~ S, M  and  ~ S, M, T+, x_{t-1}+, x_{t-2}+, dsum")
for (a, a2) in ((3, 1), (3, 2), (5, 2), (5, 3), (2, 5), (3, 6), (6, 3), (4, 1), (6, 1)):
    s = ok & (e.a1 == a) & (e.a2 == a2)
    if s.sum() < 2000: continue
    b1, R1 = lsq(Th5[s], [e.So[s], e.Mo[s]])
    b2, R2 = lsq(Th5[s], [e.So[s], e.Mo[s], To[s], np.maximum(xprev[s], 0), np.maximum(xprev2[s], 0), dsum[s]])
    print(f"  a1={a} a2={a2}: n={s.sum():6d}  R2 {R1:.3f} (bS={b1[0]:.3f}, bM={b1[1]:.3f})  ->  {R2:.3f}  coefs " + " ".join(f"{v:6.3f}" for v in b2))
