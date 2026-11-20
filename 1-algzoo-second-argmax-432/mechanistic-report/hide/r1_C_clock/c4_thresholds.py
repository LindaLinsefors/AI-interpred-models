"""C4. Thresholds of the comparators and of unit 4 from the mode expansion.

theta_j(t+1) = -(W[j,ORD] h_t[ORD] + W[j,PULSE] p_t) / w_j,   j in (1, 6, 7, 8, 4)
             = sum_real (m_n/K4) T_jn + sum_pairs 2 Re((z_n/K4) T_jn) + pulse terms,   T_jn = -K4 W[j,ORD].RIGHT_n / w_j
400k N(0,1) sequences, seed 1400.  'a' always means a1(t): the age of the max in the state h_t that sets the thresholds for x_{t+1}."""
from cl import *

UN = COMP + [4]
Tjn = {n: -K4 * (W[np.ix_(UN, ORD)] @ RIGHT[n]) / w[UN] for n in NAMES}      # (5,) per mode
print("=== [EXACT] threshold per unit of (mode coordinate / K4): T_jn (for pairs: |T|@phase, contribution = 2|T||z/K4| cos(arg z + phase)) ===")
print("  unit   " + "".join(f"{n:>16s}" for n in NAMES))
for k, j in enumerate(UN):
    print(f"  {j:4d}   " + "".join((f"{fmt(Tjn[n][k]):>16s}" if IS_COMPLEX[n] else f"{Tjn[n][k].real:16.3f}") for n in NAMES))
print("  pulse terms: -W[j,p]/w_j per unit of pulse activation (x K4 to compare):")
for k, j in enumerate(UN):
    print(f"  {j:4d}   " + "  ".join(f"p{p}: {-W[j, p] / w[j] * K4:+.3f}" for p in PULSE))

x, H, Z, st = gen(400_000, 1400)
M, S, Tq, a1, a2 = st["M1"], pos(st["M2"]), pos(st["M3"]), st["a1"], st["a2"]
m = coords(H)
th = thresholds(H)                                             # (B, T, 5), threshold for step t+1
ok = (M > 0)
ok[:, 9] = False                                               # there is no step 10
G = M - S

# exact contributions per mode
contrib = {}
for n in NAMES:
    c = m[n][..., None] / K4
    contrib[n] = 2 * np.real(c * Tjn[n]) if IS_COMPLEX[n] else np.real(c) * Tjn[n].real
contrib["pulses"] = -(H[..., PULSE] @ W[np.ix_(UN, PULSE)].T) / w[UN]
tot = sum(contrib.values())
print("\nexact decomposition check, max |sum of contributions - theta|:", np.abs(tot - th).max())

# ---------------------------------------------------------------- laws for level / runner / clock by age
print("\n=== [FIT] laws of the three slow coordinates by age a (states with M > 0, t <= 8), no intercept ===")
print("  a |      n  | level/K4 ~ lM M + lS S+  (R2) | runner/K4 ~ rM M + rS S+ + rT T+  (R2) | clock/K4 ~ alpha (M - S+)  (EV) | clock ~ alpha G + beta S + gamma T (EV)")
law = {}
def lsq(y, cols):
    X = np.column_stack(cols); c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, 1 - ((y - X @ c) ** 2).sum() / (y ** 2).sum()
for a in range(0, 9):
    s = ok & (a1 == a)
    cl_, el = lsq(m["level"][s] / K4, [M[s], S[s]])
    cr_, er = lsq(m["runner"][s] / K4, [M[s], S[s], Tq[s]])
    cr2_, er2 = lsq(m["runner"][s] / K4, [M[s], S[s]])
    cc_, ec = cfit(m["clock"][s] / K4, [G[s]])
    c3_, e3 = cfit(m["clock"][s] / K4, [G[s], S[s], Tq[s]])
    law[a] = dict(level=cl_, runner=cr2_, runner3=cr_, clock=cc_[0], clock3=c3_)
    print(f"  {a} | {int(s.sum()):7d} | {cl_[0]:+.3f} {cl_[1]:+.3f} ({el:.4f}) | {cr_[0]:+.3f} {cr_[1]:+.3f} {cr_[2]:+.3f} ({er:.4f}) | {fmt(cc_[0])} ({ec:.4f}) | {fmt(c3_[0])} {fmt(c3_[1])} {fmt(c3_[2])} ({e3:.4f})"
          f" | runner 2-term: {cr2_[0]:+.3f} M {cr2_[1]:+.3f} S ({er2:.4f})")

# ---------------------------------------------------------------- formula theta_j ~ A_j M + B_j S + (M - S) C_j cos(phi + delta_j)
print("\n=== formula  theta_j ~ A_j(a) M + B_j(a) S+ + (M - S+) C_j(a) cos(phi(a) + delta_j)   from T_jn and the fitted laws (level 2-term, runner 2-term, clock G-only) ===")
print("  delta_j = phase of T_j,clock (deg):", {j: round(float(deg(Tjn['clock'][k])), 1) for k, j in enumerate(UN)},
      "   2|T_j,clock|:", {j: round(float(2 * abs(Tjn['clock'][k])), 3) for k, j in enumerate(UN)})
print("  a | phi(a) rho(a) | per unit j in (1,6,7,8,4):  A_j, B_j, C_j, cos(phi+delta_j)  -> position of the formula threshold at S+ = 0:  A_j + C_j cos")
form = {}
for a in range(0, 9):
    L = law[a]
    row = []
    for k, j in enumerate(UN):
        Aj = L["level"][0] * Tjn["level"][k].real + L["runner"][0] * Tjn["runner"][k].real
        Bj = L["level"][1] * Tjn["level"][k].real + L["runner"][1] * Tjn["runner"][k].real
        Cj = 2 * abs(Tjn["clock"][k]) * abs(L["clock"])
        cs = np.cos(np.angle(L["clock"]) + np.angle(Tjn["clock"][k]))
        form[(a, j)] = (Aj, Bj, Cj, cs)
        row.append(f"{Aj:+.3f},{Bj:+.3f},{Cj:.3f},{cs:+.2f}->{Aj + Cj * cs:+.3f}")
    print(f"  {a} | {deg(L['clock']) % 360:6.1f} {abs(L['clock']):.3f} | " + " | ".join(row))

# ---------------------------------------------------------------- measured thresholds as positions between S and M
print("\n=== [MEASURED] threshold positions u_j = (theta_j - S+)/(M - S+) by age a: median [IQR];  states with M > 0, S+ > 0 pooled with S+ = 0 ===")
u = (th - S[..., None]) / G[..., None]
pred_law = np.zeros_like(th); pred_law3 = np.zeros_like(th)
for a in range(0, 9):
    s = ok & (a1 == a)
    for k, j in enumerate(UN):
        Aj, Bj, Cj, cs = form[(a, j)]
        pred_law[s, k] = Aj * M[s] + Bj * S[s] + G[s] * Cj * cs
    # law with the 3-term clock and runner with T
    L = law[a]
    lev = L["level"][0] * M[s] + L["level"][1] * S[s]
    run_ = L["runner3"][0] * M[s] + L["runner3"][1] * S[s] + L["runner3"][2] * Tq[s]
    zc = L["clock3"][0] * G[s] + L["clock3"][1] * S[s] + L["clock3"][2] * Tq[s]
    for k, j in enumerate(UN):
        pred_law3[s, k] = lev * Tjn["level"][k].real + run_ * Tjn["runner"][k].real + 2 * np.real(zc * Tjn["clock"][k])
u_law = (pred_law - S[..., None]) / G[..., None]
print("  a |   n    | " + " | ".join(f"unit {j}: measured median [IQR], formula median" for j in UN))
for a in range(0, 9):
    s = ok & (a1 == a)
    row = []
    for k, j in enumerate(UN):
        q = u[s][:, k]
        row.append(f"{np.median(q):+.3f} [{np.percentile(q, 25):+.3f},{np.percentile(q, 75):+.3f}]  {np.median(u_law[s][:, k]):+.3f}")
    print(f"  {a} | {int(s.sum()):6d} | " + " | ".join(row))
print("\n  same, restricted to states where the runner-up is positive and well separated: S+ > 0 and (M - S+)/M > 0.3")
for a in range(0, 9):
    s = ok & (a1 == a) & (S > 0) & (G / np.where(M > 0, M, 1) > 0.3)
    row = []
    for k, j in enumerate(UN):
        q = u[s][:, k]
        row.append(f"{np.median(q):+.3f} [{np.percentile(q, 25):+.3f},{np.percentile(q, 75):+.3f}]  {np.median(u_law[s][:, k]):+.3f}")
    print(f"  {a} | {int(s.sum()):6d} | " + " | ".join(row))
print("\n  lowest comparator threshold min_j u_j and unit-4 threshold u_4 by age: median [IQR]; share of states with min_j u_j > 0.25 (a new runner-up up to a quarter of the way to M clips nothing); share with u_4 < 0.9")
for a in range(0, 9):
    s = ok & (a1 == a)
    lo = u[s][:, :4].min(1); u4 = u[s][:, 4]
    print(f"  a={a}: min comparator {np.median(lo):+.3f} [{np.percentile(lo, 25):+.3f},{np.percentile(lo, 75):+.3f}]  (lowest is unit {COMP[np.bincount(u[s][:, :4].argmin(1), minlength=4).argmax()]} in {np.bincount(u[s][:, :4].argmin(1), minlength=4).max() / s.sum():.2f})"
          f"   share(min > 0.25) = {np.mean(lo > 0.25):.3f}   share(min < -0.1) = {np.mean(lo < -0.1):.3f} | unit 4: {np.median(u4):+.3f} [{np.percentile(u4, 25):+.3f},{np.percentile(u4, 75):+.3f}]  share(u_4 < 0.9) = {np.mean(u4 < 0.9):.3f}  share(u_4 > 1) = {np.mean(u4 > 1):.3f}")

# ---------------------------------------------------------------- variance accounting
print("\n=== how much of each threshold do level + runner + clock explain?  (states with M > 0, t <= 8) ===")
def r2c(y, p):          # centred R2 of prediction p for y (no refitting)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
sel = ok & (a1 >= 1)
lrc = contrib["level"] + contrib["runner"] + contrib["clock"]
preds = (("level only (exact coordinate)", contrib["level"]), ("level + runner (exact)", contrib["level"] + contrib["runner"]),
         ("level + runner + clock (exact coordinates)", lrc), ("  + short + trans", lrc + contrib["short"] + contrib["trans"]),
         ("  + all modes (no pulses)", tot - contrib["pulses"]),
         ("formula A M + B S + (M-S) C cos (laws by age)", pred_law), ("laws with 3-term clock and runner incl. T", pred_law3))
print("  (T1) target theta_j (raw), a >= 1.  columns: units 1, 6, 7, 8, 4;  centred R2, no refitting")
for label, p_ in preds:
    print(f"    {label:48s} " + "  ".join(f"{r2c(th[sel][:, k], p_[sel][:, k]):.4f}" for k in range(5)))
print("  (T2) target theta_j - S+ (height of the threshold above the runner-up, raw), a >= 1")
Sx = S[..., None]
for label, p_ in preds[2:]:
    print(f"    {label:48s} " + "  ".join(f"{r2c((th - Sx)[sel][:, k], (p_ - Sx)[sel][:, k]):.4f}" for k in range(5)))
print("  (T3) target position u_j = (theta_j - S+)/(M - S+), states with a >= 1, M > 0.5 and (M - S+)/M > 0.3:  centred R2  |  median |error| of the predicted position")
sel3 = sel & (M > 0.5) & (G / np.where(M > 0, M, 1) > 0.3)
up = lambda arr: (arr - Sx) / G[..., None]
for label, p_ in preds[2:]:
    e = np.abs(up(p_)[sel3] - u[sel3])
    print(f"    {label:48s} " + "  ".join(f"{r2c(u[sel3][:, k], up(p_)[sel3][:, k]):.4f}" for k in range(5)) + "   |   " + "  ".join(f"{np.median(e[:, k]):.3f}" for k in range(5)))
print(f"    (n = {int(sel3.sum())} states; sd of the measured positions: " + "  ".join(f"{u[sel3][:, k].std():.3f}" for k in range(5)) + ")")
print("  share of var(theta_j - S+) carried by each exact contribution, a >= 1: cov(contribution, target)/var(target); with the '-S+' row the shares sum to 1")
y = (th - Sx)[sel]
for n in list(NAMES) + ["pulses"]:
    c = contrib[n][sel]
    print(f"    {n:7s} " + "  ".join(f"{np.mean((c[:, k] - c[:, k].mean()) * (y[:, k] - y[:, k].mean())) / y[:, k].var():+.4f}" for k in range(5)))
print("    -S+     " + "  ".join(f"{np.mean((-S[sel] + S[sel].mean()) * (y[:, k] - y[:, k].mean())) / y[:, k].var():+.4f}" for k in range(5)))
print("\n  by age a: centred R2 of theta_j - S+ from level+runner+clock exact coordinates | from the formula with laws | rms of the part carried by other modes and pulses, relative to rms(M - S+)")
for a in range(0, 9):
    s = ok & (a1 == a)
    print(f"    a={a}: " + "  ".join(f"{r2c((th - Sx)[s][:, k], (lrc - Sx)[s][:, k]):.3f}" for k in range(5)) + "   |   " + "  ".join(f"{r2c((th - Sx)[s][:, k], (pred_law - Sx)[s][:, k]):.3f}" for k in range(5))
          + "   |   " + "  ".join(f"{np.sqrt(np.mean((th - lrc)[s][:, k] ** 2) / np.mean(G[s] ** 2)):.3f}" for k in range(5)))
print("\n  median exact contributions to theta_j / M by age (level | runner | clock | h2 | alt | h3 | short | trans | echo | pulses), unit by unit")
for k, j in enumerate(UN):
    print(f"   unit {j}")
    for a in range(0, 9):
        s = ok & (a1 == a)
        print(f"     a={a}: " + "  ".join(f"{np.median(contrib[n][s][:, k] / M[s]):+.3f}" for n in list(NAMES) + ["pulses"]) + f"   total {np.median(th[s][:, k] / M[s]):+.3f}")

np.savez(HERE / "c4_thresholds.npz", med_u=np.array([[np.median(u[ok & (a1 == a)][:, k]) for k in range(5)] for a in range(9)]))
