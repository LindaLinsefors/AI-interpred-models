"""H4 part 1-3.  Unit 5 as a fifth comparator.

With all 13 ordinary units ON at step t (no comparator clipped, no wave unit clipped):
    z5(t+1) = w5 x_{t+1} + c5 (x_t - Theta5(t)),    c5 = W[5,ORD].w[ORD] = 18.009
    c5 * Theta5(t) = - W[5,ORD].(A h_{t-1}[ORD] + W[ORD,PULSE] h_{t-1}[PULSE]) - W[5,PULSE].h_t[PULSE]
In mode coordinates m(t-1) = LEFT.h_{t-1}[ORD]:
    W[5,ORD].A h_{t-1}[ORD] = sum_k g_k m_k(t-1)   (2 Re(.) for complex pairs),  g_k = lam_k * W[5,ORD].RIGHT_k
"""
from hcommon import *

c5 = W[5, ORD] @ w[ORD]
g = {n: LAM[n] * (W[5, ORD] @ RIGHT[n]) for n in NAMES}
print("[EXACT] c5 =", round(c5, 4))
print("[EXACT] g_k / c5 (threshold shift per unit mode coordinate; minus sign applied: Theta5 = -sum g_k m_k / c5 + pulses):")
for n in NAMES:
    v = -g[n] / c5
    print(f"   {n:7s}: " + (f"2|.| = {2 * abs(v):.5f} @ {np.degrees(np.angle(v)):7.1f} deg" if IS_COMPLEX[n] else f"{v.real:9.5f}"))
print("[EXACT] pulse terms of c5*Theta5: -W[5,ORD].W[ORD,PULSE] (h_{t-1}[4,5,9]) =", np.round(-W[5, ORD] @ W[np.ix_(ORD, PULSE)], 3),
      ";  -W[5,PULSE] (h_t[4,5,9]) =", np.round(-W[5, PULSE], 3))

cols = {}
for seed in (1007, 1008, 1009):
    x = sample(400_000, seed)
    e = Ev(x)
    m = coords(e.Hprev)                                  # mode coordinates at t-1, aligned with t
    pre0 = e.Hprev @ W.T                                  # W h_{t-1}: pre-activations at step t for x_t = 0
    lin_x = -pre0[:, :, ORD] @ W[5, ORD] - e.H[:, :, PULSE] @ W[5, PULSE]        # exact c5*Theta5 (uses the actual pulses at t)
    Th5x = lin_x / c5
    # state-only version: unit 4 dropped (it is silent for inputs below the max), units 5, 9 at step t evaluated at x_t = 0
    pstate = -(e.Hprev[:, :, PULSE] @ W[np.ix_(ORD, PULSE)].T) @ W[5, ORD] - W[5, 5] * np.maximum(pre0[:, :, 5], 0) - W[5, 9] * np.maximum(pre0[:, :, 9], 0)
    Th5 = (-(e.Hprev[:, :, ORD] @ A.T) @ W[5, ORD] + pstate) / c5
    allon = e.G[:, :, ORD].all(-1)
    # check: when all 13 are on at t, z5(t+1) = w5 x_{t+1} + c5 (x_t - Theta5x)   (exact)
    pred = w[5] * x[:, 1:] + c5 * (x[:, :-1] - Th5x[:, :-1])
    err = np.abs(pred - e.Z[:, 1:, 5])[allon[:, :-1]].max()
    off4 = ~e.G[:, :, 4]
    print(f"seed {seed}: all-13-on steps: {allon[:, :-1].mean():.3f} of steps t<=8; max |z5(t+1) - formula| = {err:.2e};  "
          f"|Theta5(state only) - exact| on unit-4-off steps: median {np.median(np.abs(Th5 - Th5x)[off4]):.4f}, 99% {np.percentile(np.abs(Th5 - Th5x)[off4], 99):.4f}")
    ok = (e.tt >= 2) & (e.tt <= 8) & (e.So > 0)
    b, t = np.nonzero(ok)
    d = dict(a=e.a1[b, t], a2=e.a2[b, t], t=t, M=e.Mo[b, t], S=e.So[b, t], T=np.maximum(e.To[b, t], 0), x=x[b, t], xn=x[b, t + 1], Th5=Th5[b, t],
             on5n=e.G[b, t + 1, 5], z5n=e.Z[b, t + 1, 5], allon=allon[b, t], on4=e.G[b, t, 4],
             ncomp_off=(~e.G[b, t][:, COMP]).sum(1), nwave_off=(~e.G[b, t][:, WAVE]).sum(1),
             pulse_part=pstate[b, t] / c5, Th5x=Th5x[b, t],
             xprev=x[b, t - 1])
    for n in NAMES:
        mm = m[n][b, t]
        d[f"part_{n}"] = (-(2 * np.real(g[n] * mm) if IS_COMPLEX[n] else np.real(g[n]) * mm.real)) / c5
        d[f"m_{n}"] = mm
    for j in COMP:
        d[f"th{j}"] = e.th[j][b, t]
    for k, v in d.items():
        cols.setdefault(k, []).append(v)
    del e, m
d = {k: np.concatenate(v) for k, v in cols.items()}
n = len(d["a"]); print("steps with 2<=t<=8 and S_old>0:", n)
chk = sum(d[f"part_{k}"] for k in NAMES) + d["pulse_part"]
print("mode decomposition of Theta5 exact: max |sum of parts - Theta5| =", np.abs(chk - d["Th5"]).max())

M, S = d["M"], d["S"]
print("\n=== Theta5 ~ bS S_old + bM M (no intercept) by age of max; rho5 = (Theta5 - S)/(M - S) ===")
print("  a  |   bS      bM     R2    | with T+ (3rd largest): bS, bM, bT, R2 | median Theta5/S for gap (M-S)/M in [0,.2) [.2,.5) [.5,.8) [.8,1]")
for a in range(1, 9):
    s = d["a"] == a
    bb, R2 = lsq(d["Th5"][s], [S[s], M[s]])
    b3, R3 = lsq(d["Th5"][s], [S[s], M[s], d["T"][s]])
    gm = (M[s] - S[s]) / M[s]
    med = [np.median((d["Th5"][s] / S[s])[(gm >= lo) & (gm < hi)]) for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01))]
    print(f"  {a}  | {bb[0]:6.3f}  {bb[1]:6.3f}  {R2:.4f} | {b3[0]:6.3f} {b3[1]:6.3f} {b3[2]:6.3f} {R3:.4f} | " + "  ".join(f"{v:6.3f}" for v in med))

print("\n=== same split by a2 < a (runner-up younger than max) vs a2 > a (runner-up is a former max) ===")
for a in range(1, 9):
    row = f"  a={a}: "
    for lab, cond in (("S older", d["a2"] > d["a"]), ("S younger", d["a2"] < d["a"])):
        s = (d["a"] == a) & cond
        if s.sum() < 2000: row += f"{lab}: .                          | "; continue
        bb, R2 = lsq(d["Th5"][s], [S[s], M[s]])
        row += f"{lab}: bS={bb[0]:6.3f} bM={bb[1]:6.3f} R2={R2:.4f} | "
    print(row)

print("\n=== which modes make Theta5: per age, each part regressed on (S, M): bS, bM  (parts sum exactly to Theta5) ===")
print("  a  | " + " | ".join(f"{k:^13s}" for k in NAMES + ["pulses"]))
for a in range(1, 9):
    s = d["a"] == a
    row = ""
    for k in NAMES + ["pulses"]:
        v = d["pulse_part"][s] if k == "pulses" else d[f"part_{k}"][s]
        bb, _ = lsq(v, [S[s], M[s]])
        row += f"{bb[0]:6.3f} {bb[1]:6.3f} | "
    print(f"  {a}  | {row}")

print("\n=== mode coordinates at t-1 in units of K: m_k/K ~ bS S + bM M by age of max (a-1 = age at t-1) ===")
for k in ("level", "runner", "trans"):
    print(f"  {k}: " + "  ".join("a={}: {:.3f} S + {:.3f} M (R2 {:.3f})".format(a, *lsq(d[f'm_{k}'][d['a'] == a].real / K, [S[d['a'] == a], M[d['a'] == a]])[0], lsq(d[f'm_{k}'][d['a'] == a].real / K, [S[d['a'] == a], M[d['a'] == a]])[1]) for a in range(1, 9)))

# --- P(unit 5 on at t+1) against x_t / S ---
below = (d["x"] < S) & ~d["on4"]
ratio = d["x"] / S
gm = (M - S) / M
RB = [-np.inf, 0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print("\n=== P(unit 5 ON at t+1 | x_t < S_old) against x_t/S_old ===")
hdr = "".join(f" {('<0' if i == 0 else f'{RB[i]:.1f}-{RB[i+1]:.1f}'):>8s}" for i in range(len(RB) - 1))
print(" split by (M-S)/M:\n   gap bin   |" + hdr)
for lo, hi in ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, .7), (.7, 1.01)):
    row = ""
    for i in range(len(RB) - 1):
        s = below & (gm >= lo) & (gm < hi) & (ratio >= RB[i]) & (ratio < RB[i + 1])
        row += f" {d['on5n'][s].mean():8.3f}" if s.sum() > 300 else "      .  "
    print(f"   {lo:.2f}-{hi:.2f} |{row}")
print(" split by age of max:\n   a         |" + hdr)
for a in range(1, 9):
    row = ""
    for i in range(len(RB) - 1):
        s = below & (d["a"] == a) & (ratio >= RB[i]) & (ratio < RB[i + 1])
        row += f" {d['on5n'][s].mean():8.3f}" if s.sum() > 300 else "      .  "
    print(f"   a={a}       |{row}")
print(" all        |" + "".join(f" {d['on5n'][below & (ratio >= RB[i]) & (ratio < RB[i + 1])].mean():8.3f}" for i in range(len(RB) - 1)))
print(" share of below-S inputs in each ratio bin:" + "".join(f" {(below & (ratio >= RB[i]) & (ratio < RB[i + 1])).sum() / below.sum():8.3f}" for i in range(len(RB) - 1)))

print("\n=== the ratio at which unit 5 switches on: among below-S inputs, 50% point of P(on) vs x_t/S, by gap bin and age (logistic-free: median of Theta5eff/S) ===")
# effective threshold on x_t including clipped units: solve z5(t+1)=0 is piecewise; use the all-on formula where valid
s0 = below & d["allon"]
print(f"  below-S inputs with all 13 ordinary units on at t: {s0.sum() / below.sum():.3f} of below-S inputs")
print("  agreement of the rule 'unit 5 on iff x_t > Theta5x + (0.281/18.009) x_{t+1}' (exact threshold) on those:", np.mean((d["x"][s0] > d["Th5x"][s0] - w[5] / c5 * d["xn"][s0]) == d["on5n"][s0]))
print("  agreement of the rule 'unit 5 on iff x_t > Theta5x' on ALL below-S inputs:", np.mean((d["x"][below] > d["Th5x"][below]) == d["on5n"][below]))
print("  agreement of the rule 'unit 5 on iff x_t > Theta5 (state only)' on ALL below-S inputs:", np.mean((d["x"][below] > d["Th5"][below]) == d["on5n"][below]))
print("  agreement of the rule 'unit 5 on iff x_t > 0.75 S_old' on ALL below-S inputs:", np.mean((d["x"][below] > 0.75 * S[below]) == d["on5n"][below]))
print("  median Theta5/S [25%,75%] by gap bin (rows) and age (cols 1..8), all steps with S_old>0:")
for lo, hi in ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, .7), (.7, 1.01)):
    row = ""
    for a in range(1, 9):
        s = (d["a"] == a) & (gm >= lo) & (gm < hi)
        row += f" {np.median(d['Th5'][s] / S[s]):6.2f}" if s.sum() > 300 else "    .  "
    print(f"   gap {lo:.2f}-{hi:.2f}: {row}")

print("\n=== comparison of all five thresholds: median rho = (theta - S)/(M - S) by age, steps with gap/M > 0.2 ===")
print("  a  |  unit 5 (Theta5) | unit 1 | unit 6 | unit 7 | unit 8")
for a in range(1, 9):
    s = (d["a"] == a) & (gm > 0.2)
    print(f"  {a}  |  {np.median((d['Th5'][s] - S[s]) / (M[s] - S[s])):6.3f} [{np.percentile((d['Th5'][s] - S[s]) / (M[s] - S[s]), 25):6.3f},{np.percentile((d['Th5'][s] - S[s]) / (M[s] - S[s]), 75):6.3f}] | "
          + " | ".join(f"{np.median((d[f'th{j}'][s] - S[s]) / (M[s] - S[s])):6.3f}" for j in COMP))
