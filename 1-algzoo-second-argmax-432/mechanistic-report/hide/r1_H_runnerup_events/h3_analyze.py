"""H3 analysis of the cached counterfactual N events (h3_event.npz)."""
from hcommon import *
from h3_event import REL, GROUPS, load_events

d = load_events()
n = len(d["t"])
xs, S, M, a, t = d["x"] - d["S"], d["S"], d["M"], d["a"], d["t"]
print(f"N events: {n};  counterfactual (x_t=0) itself quiet in {d['cf_quiet'].mean():.3f}")
quiet2 = d["q2"] < 0            # the two followers are negative
quietall = d["qmax"] < 0        # all followers negative

# ---------- A. mode coordinates at t+2 ----------
print("\n=== A. change of the mode coordinates at t+2 caused by the event, in units of K = 10.157 ===")
print("    Delta m_k / K ~ alpha (x_t - S_old) + beta S_old + gamma M   (events with t<=7 and x_{t+1}, x_{t+2} < 0; no intercept)")
for nme in NAMES:
    cplx = IS_COMPLEX[nme]
    print(f"  mode {nme}" + (f"  (free rotation {np.degrees(np.angle(LAM[nme])):.1f} deg/step; phases are at t+2)" if cplx else ""))
    for A_ in range(1, 8):
        s = (a == A_) & (t <= 7) & quiet2
        if s.sum() < 300: continue
        y = d[f"dm_{nme}"][s] / K
        b, R2 = lsq(y.astype(complex) if cplx else y.real, [xs[s].astype(complex) if cplx else xs[s], S[s], M[s]])
        f = (lambda c: f"{abs(c):6.3f}@{np.degrees(np.angle(c)):6.1f}") if cplx else (lambda c: f"{c.real:7.3f}")
        # single-regressor fit: alpha only
        b1, R1 = lsq(y.astype(complex) if cplx else y.real, [xs[s].astype(complex) if cplx else xs[s]])
        print(f"    a={A_}: alpha={f(b[0])}  beta={f(b[1])}  gamma={f(b[2])}  R2={R2:.3f} | alpha-only fit: {f(b1[0])} R2={R1:.3f}  n={s.sum()}")

# ---------- B. logit kernel ----------
print("\n=== B. logit change caused by the event (actual minus x_t:=0), all followers; regression on (x_t - S_old, S_old, M) ===")
def reg(y, s):
    ok = s & np.isfinite(y)
    b, R2 = lsq(y[ok], [xs[ok], S[ok], M[ok]])
    return b, R2, ok.sum()

for where, lab in (("r0", "own position t"), ("p2", "position of the OLD runner-up"), ("p1", "position of the max"), ("r1", "position t+1"), ("r-1", "position t-1")):
    print(f"  total logit change at the {lab}:  alpha, beta, gamma, R2 by age")
    for A_ in range(1, 9):
        b, R2, k = reg(d[f"total_{where}"], a == A_)
        print(f"    a={A_}: {b[0]:8.2f} {b[1]:8.2f} {b[2]:8.2f}   R2={R2:.3f}  n={k}")

print("\n=== B2. mean logit change / M by relative position (rows: age; all followers) ===")
print("   a  | " + " ".join(f"  d{r:+d}  " for r in REL) + " |   p_max   p_oldS")
for A_ in range(1, 9):
    s = a == A_
    row = " ".join(f"{np.nanmean(d[f'total_r{r}'][s] / M[s]):7.2f}" for r in REL)
    print(f"   {A_}  | {row} | {np.nanmean(d['total_p1'][s] / M[s]):7.2f} {np.nanmean(d['total_p2'][s] / M[s]):7.2f}")

print("\n=== B3. the same, split by source: mean (logit change / M) at own position t | at old runner-up | at the max ===")
print("   a  | " + " | ".join(f"{g:^22s}" for g in GROUPS))
for A_ in range(1, 9):
    s = a == A_
    print(f"   {A_}  | " + " | ".join(f"{np.nanmean(d[f'{g}_r0'][s] / M[s]):6.2f} {np.nanmean(d[f'{g}_p2'][s] / M[s]):6.2f} {np.nanmean(d[f'{g}_p1'][s] / M[s]):6.2f}  " for g in GROUPS))

print("\n=== B4. regression coefficient alpha (per unit of x_t at fixed S_old, M) of each source, at own position | old runner-up | max ===")
print("   a  | " + " | ".join(f"{g:^22s}" for g in GROUPS))
for A_ in range(1, 9):
    row = ""
    for g in GROUPS:
        row += " ".join(f"{reg(d[f'{g}_{wh}'], a == A_)[0][0]:6.2f}" for wh in ("r0", "p2", "p1")) + "   | "
    print(f"   {A_}  | {row}")

# ---------- C. the decision between position t and the old runner-up ----------
print("\n=== C. margin of position t over the old runner-up's position at the end:  actual, counterfactual, and the change ===")
m1 = d["L1_t"] - d["L1_p2"]; m0 = d["L0_t"] - d["L0_p2"]
print("   regression of each on (x_t - S_old, S_old, M), by age (all followers)")
for A_ in range(1, 9):
    s = a == A_
    b1, R1 = lsq(m1[s], [xs[s], S[s], M[s]]); b0, R0 = lsq(m0[s], [xs[s], S[s], M[s]]); bd, Rd = lsq((m1 - m0)[s], [xs[s], S[s], M[s]])
    print(f"    a={A_}: actual: {b1[0]:7.2f} {b1[1]:7.2f} {b1[2]:7.2f} (R2 {R1:.2f}) | x_t:=0: {b0[0]:7.2f} {b0[1]:7.2f} {b0[2]:7.2f} (R2 {R0:.2f}) | change: {bd[0]:7.2f} {bd[1]:7.2f} {bd[2]:7.2f} (R2 {Rd:.2f})"
          f" | P(margin>0) = {np.mean(m1[s] > 0):.3f}")
print("   quiet followers only (all later inputs < 0):")
for A_ in range(1, 9):
    s = (a == A_) & quietall
    if s.sum() < 300: continue
    b1, R1 = lsq(m1[s], [xs[s], S[s], M[s]]); b0, R0 = lsq(m0[s], [xs[s], S[s], M[s]]); bd, Rd = lsq((m1 - m0)[s], [xs[s], S[s], M[s]])
    print(f"    a={A_}: actual: {b1[0]:7.2f} {b1[1]:7.2f} {b1[2]:7.2f} (R2 {R1:.2f}) | x_t:=0: {b0[0]:7.2f} {b0[1]:7.2f} {b0[2]:7.2f} (R2 {R0:.2f}) | change: {bd[0]:7.2f} {bd[1]:7.2f} {bd[2]:7.2f} (R2 {Rd:.2f})"
          f" | P(margin>0) = {np.mean(m1[s] > 0):.3f}  n={s.sum()}")

print("\n=== D. size of the unit-5 pulse: h_{t+1}[5] ~ (x_t - S_old, S_old, M) by age; counterfactual h5 (x_t = 0) mean ===")
for A_ in range(1, 9):
    s = a == A_
    b, R2 = lsq(d["h5"][s], [xs[s], S[s], M[s]])
    print(f"    a={A_}: h5 = {b[0]:6.2f} (x-S) + {b[1]:6.2f} S + {b[2]:6.2f} M  (R2 {R2:.3f});  mean h5 = {d['h5'][s].mean():.2f}, mean h5 with x_t=0: {d['h5cf'][s].mean():.3f}")
