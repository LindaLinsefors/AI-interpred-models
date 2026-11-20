"""H2: exact piecewise form of unit 5's pre-activation, and an out-of-sample test of the reduced formula.

Exact (every step):   z5(t+1) = w5 x_{t+1} + c5 (x_t - Theta5x(t)) + sum_{i in ORD} W[5,i] c_t[i]
   c_t[i] = ReLU(-z_t[i]) (clip correction);  for a comparator j: W[5,j] c_t[j] = kappa_j (x_t - theta_j)_+ .
Reduced:  z5(t+1) - w5 x_{t+1}  ~  c5 (x_t - Th5) - 10.462 (x_t - th8)_+ - 7.813 (x_t - th7)_+
   with the per-age laws  Th5 = bS S + bM M,  th_j = bjS S + bjM M  fitted on one seed and tested on another.
"""
from hcommon import *

c5 = W[5, ORD] @ w[ORD]
kap = {j: W[5, j] * abs(w[j]) for j in COMP}


def gather(seed, n=400_000):
    x = sample(n, seed)
    e = Ev(x)
    pre0 = e.Hprev @ W.T
    Th5x = (-pre0[:, :, ORD] @ W[5, ORD] - e.H[:, :, PULSE] @ W[5, PULSE]) / c5
    pstate = -(e.Hprev[:, :, PULSE] @ W[np.ix_(ORD, PULSE)].T) @ W[5, ORD] - W[5, 5] * np.maximum(pre0[:, :, 5], 0) - W[5, 9] * np.maximum(pre0[:, :, 9], 0)
    Th5 = (-(e.Hprev[:, :, ORD] @ A.T) @ W[5, ORD] + pstate) / c5
    clip = np.maximum(-e.Z[:, :, ORD], 0)
    exact = w[5] * x[:, 1:] + c5 * (x[:, :-1] - Th5x[:, :-1]) + clip[:, :-1] @ W[5, ORD]
    err = np.abs(exact - e.Z[:, 1:, 5]).max()
    comp_only = w[5] * x[:, 1:] + c5 * (x[:, :-1] - Th5x[:, :-1]) + sum(kap[j] * np.maximum(x[:, :-1] - e.th[j][:, :-1], 0) for j in COMP)
    return x, e, Th5, err, np.abs(comp_only - e.Z[:, 1:, 5])


x, e, Th5, err, werr = gather(1023)
print(f"[EXACT] piecewise identity, seed 1023: max |z5(t+1) - formula| = {err:.2e}")
print(f"        leaving out the wave-unit clip terms: |error| median {np.median(werr):.4f}, 90% {np.percentile(werr, 90):.3f}, 99% {np.percentile(werr, 99):.3f}  (z5 itself has sd {e.Z[:, 1:, 5].std():.2f})")
print("[EXACT] slopes of z5(t+1) in x_t: all comparators on", round(c5, 3), "| 8 clipped", round(c5 + kap[8], 3), "| 7 clipped", round(c5 + kap[7], 3), "| 7 and 8 clipped", round(c5 + kap[7] + kap[8], 3),
      "| all four clipped", round(c5 + sum(kap.values()), 3))

# per-age laws from seed 1023 (all steps t>=2 with S_old>0; thresholds are functions of the state only)
ok = (e.tt >= 2) & (e.tt <= 8) & (e.So > 0)
law = {}
for a in range(1, 9):
    s = ok & (e.a1 == a)
    law[a] = {k: lsq(v[s], [e.So[s], e.Mo[s]])[0] for k, v in (("Th5", Th5), ("th7", e.th[7]), ("th8", e.th[8]))}
print("\nper-age laws fitted on seed 1023:  Theta5 = bS S + bM M | theta_7 | theta_8")
for a in range(1, 9):
    print(f"   a={a}: " + " | ".join(f"{law[a][k][0]:6.3f} S + {law[a][k][1]:6.3f} M" for k in ("Th5", "th7", "th8")))

# test on seed 1024
x, e, Th5, err, _ = gather(1024)
ok = (e.tt >= 2) & (e.tt <= 8) & (e.So > 0)
isN = ok & (x > e.So) & (x < e.Mo) & ~e.G[:, :, 4]
b, t = np.nonzero(isN)
a, S, M, xt = e.a1[b, t], e.So[b, t], e.Mo[b, t], x[b, t]
y = e.Z[b, t + 1, 5] - w[5] * x[b, t + 1]
pred = np.zeros(len(b)); pred_true = np.zeros(len(b))
for A_ in range(1, 9):
    s = a == A_
    L = law[A_]
    f = lambda k: L[k][0] * S[s] + L[k][1] * M[s]
    pred[s] = c5 * (xt[s] - f("Th5")) + kap[8] * np.maximum(xt[s] - f("th8"), 0) + kap[7] * np.maximum(xt[s] - f("th7"), 0)
# the same formula with the TRUE thresholds of each event (isolates the error of the threshold laws from the neglected terms)
pred_true = c5 * (xt - Th5[b, t]) + kap[8] * np.maximum(xt - e.th[8][b, t], 0) + kap[7] * np.maximum(xt - e.th[7][b, t], 0)
pat = e.G[b, t, 7].astype(int) * 2 + e.G[b, t, 8].astype(int)
PN = {3: "7on 8on", 2: "7on 8off", 1: "7off 8on", 0: "7off 8off"}
r2 = lambda yy, pp: 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
print(f"\nout-of-sample test on seed 1024, N events (unit 4 silent, 2<=t<=8, S_old>0), n = {len(b)}")
print(f"  all ages: R2 of the reduced formula = {r2(y, pred):.4f};  with the true thresholds of each event (only wave clips, comparators 1, 6 neglected) = {r2(y, pred_true):.4f}")
print("  a  | R2 reduced formula | R2 with true thresholds | by pattern of (7, 8) at t: share, R2 reduced, mean z5 actual / predicted")
for A_ in range(1, 9):
    s = a == A_
    row = ""
    for p in (3, 2, 1, 0):
        ss = s & (pat == p)
        if ss.sum() < 500: continue
        row += f" [{PN[p]}: {ss.sum() / s.sum():.2f}, R2 {r2(y[ss], pred[ss]):.3f}, {y[ss].mean():5.2f} / {pred[ss].mean():5.2f}]"
    print(f"  {A_}  |  {r2(y[s], pred[s]):.4f}           |  {r2(y[s], pred_true[s]):.4f}                |{row}")
print(f"  sign agreement (unit 5 on at t+1) of the reduced formula: {np.mean((pred + w[5] * x[b, t + 1] > 0) == (e.Z[b, t + 1, 5] > 0)):.4f}")

# --- unit 9 after a new max vs after an N event ---
print("\nunit 9 at t+1 by the class of x_t (2<=t<=8, S_old>0), seed 1024:")
cls = {"new max (x_t > M_old)": ok & (x > e.Mo), "N, unit 4 silent": isN, "N-range, unit 4 misfired": ok & (x > e.So) & (x < e.Mo) & e.G[:, :, 4], "below S_old": ok & (x < e.So)}
for k, m in cls.items():
    bb, tt_ = np.nonzero(m)
    print(f"   {k:28s}: P(unit 9 on at t+1) = {e.G[bb, tt_ + 1, 9].mean():.4f}   P(unit 5 on at t+1) = {e.G[bb, tt_ + 1, 5].mean():.4f}   n = {len(bb)}")
print("   new max, by age of the previous max a: P(unit 9 on) =", " ".join(f"{e.G[:, 1:, 9][(cls['new max (x_t > M_old)'] & (e.a1 == A_))[:, :-1]].mean():.3f}" for A_ in range(1, 9)))
# among N events: unit 9 fires iff comparator 7 is (nearly) clipped: condition x_t > theta_7 - margin
z9 = e.Z[b, t + 1, 9]
d7 = (xt - e.th[7][b, t]) / M
print("   N events: P(unit 9 on at t+1) by (x_t - theta_7)/M:  " + "  ".join(f"[{lo:+.2f},{hi:+.2f}): {np.mean(z9[(d7 >= lo) & (d7 < hi)] > 0):.3f} (n={((d7 >= lo) & (d7 < hi)).sum()})" for lo, hi in ((-9, -0.1), (-0.1, -0.03), (-0.03, 0), (0, 0.05), (0.05, 9))))

# --- composed coefficients (from the weights and the per-age threshold laws) vs a direct regression on the test seed ---
print("\ncomposed  z5(t+1) - w5 x_{t+1} ~ cx x_t + cS S_old + cM M  versus a direct least-squares fit, per (age, pattern of comparators 7, 8), seed 1024")
print("  cx = c5 + kappa_8 [8 off] + kappa_7 [7 off];  cS = -c5 bS - kappa_8 b8S [8 off] - kappa_7 b7S [7 off];  cM likewise")
print("  a  pattern    share |  cx composed / fit |  cS composed / fit |  cM composed / fit | R2 of the fit")
for A_ in range(1, 9):
    for p in (3, 2, 1, 0):
        ss = (a == A_) & (pat == p)
        if ss.sum() < 1500: continue
        on7, on8 = p >> 1, p & 1
        L = law[A_]
        cx = c5 + kap[8] * (1 - on8) + kap[7] * (1 - on7)
        cS = -c5 * L["Th5"][0] - kap[8] * L["th8"][0] * (1 - on8) - kap[7] * L["th7"][0] * (1 - on7)
        cM = -c5 * L["Th5"][1] - kap[8] * L["th8"][1] * (1 - on8) - kap[7] * L["th7"][1] * (1 - on7)
        bb, R2 = lsq(y[ss], [xt[ss], S[ss], M[ss]])
        print(f"  {A_}  {PN[p]:9s}  {ss.sum() / (a == A_).sum():.2f}  | {cx:7.2f} / {bb[0]:7.2f}  | {cS:7.2f} / {bb[1]:7.2f}  | {cM:7.2f} / {bb[2]:7.2f}  | {R2:.3f}")
