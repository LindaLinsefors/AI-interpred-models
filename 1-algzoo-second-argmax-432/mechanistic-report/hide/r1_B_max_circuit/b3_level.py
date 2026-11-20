"""B3. Level bookkeeping.
  level(t) = lam*level(t-1) + L.W[ORD,4] h4(t-1) + L.W[ORD,5] h5(t-1) + L.W[ORD,9] h9(t-1) + L.w[ORD] x_t + L.clip_t
  clip_t = h_t[ORD] - z_t[ORD] = max(-z_t[ORD], 0)   (what the ReLU adds back on the ordinary units that are off)
Data: N(0,1), seeds 1030, 1031 (400k each).
"""
from blib import *

L = LEFT["level"].real
lam = LAM["level"].real
k4, k5, k9 = (L @ W[ORD, q] for q in PULSE)
kx = L @ w[ORD]
CI = [OI[u] for u in COMP]
WI = [OI[u] for u in WAVE]
print("=== [EXACT] coefficients of the level update ===")
print(f"  lam_level = {lam:.5f};  L.W[ORD,4] = {k4:.4f};  L.W[ORD,5] = {k5:.4f};  L.W[ORD,9] = {k9:.4f};  L.w[ORD] = {kx:.4f}")
print("  L_i (clip coefficient per ordinary unit) and L_i*w_i (share of L.w[ORD]):")
for i, u in enumerate(ORD):
    print(f"    unit {u:2d}: L = {L[i]:8.4f}   L*w = {L[i]*w[u]:8.4f}" + ("   <- comparator" if u in COMP else ""))
print(f"  sum over comparators of L*w = {sum(L[i]*w[ORD[i]] for i in CI):.4f};  over wave units = {sum(L[i]*w[ORD[i]] for i in WI):.4f}")
print(f"  L_level . RIGHT_echo = {abs(LEFT['level'] @ RIGHT['echo']):.2e} (exactly 0 by biorthogonality; w[ORD] is 99.8% echo, which is why L.w is small)")


def level_terms(x, H, Z):
    """All terms of the exact update for steps t = 0..T-1 (previous state = 0 at t = 0). Shapes (B, T)."""
    lev = H[:, :, ORD] @ L
    Hprev = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
    levprev = Hprev[:, :, ORD] @ L
    clip = np.maximum(-Z[:, :, ORD], 0)
    T = {
        "growth": (lam - 1) * levprev,
        "u4": k4 * Hprev[:, :, 4],
        "u5": k5 * Hprev[:, :, 5],
        "u9": k9 * Hprev[:, :, 9],
        "x": kx * x,
        "clipC": clip[:, :, CI] @ L[CI],
        "clipW": clip[:, :, WI] @ L[WI],
    }
    assert np.allclose(lev - levprev, sum(T.values()), atol=1e-8)
    return lev, levprev, T, clip


print("\n=== [EXACT] isolated record: x = (1, b, b, ...), level/10.157 and the update terms/10.157 ===")
for b in (0.0, -0.5, -1.0):
    xs = np.r_[1.0, np.full(9, b)][None]
    H, Z = run(xs, return_pre=True)
    lev, levprev, T, clip = level_terms(xs, H, Z)
    print(f"  background b = {b}")
    print("   t | level/10.157 | " + " | ".join(f"{k:>7s}" for k in T) + " |  h4/10.157  h5/10.157  h9/10.157 | ordinary units off")
    for t in range(10):
        off = [u for u in ORD if Z[0, t, u] <= 0]
        print(f"   {t} | {lev[0,t]/W4:12.4f} | " + " | ".join(f"{T[k][0,t]/W4:7.4f}" for k in T)
              + f" | {H[0,t,4]/W4:10.4f} {H[0,t,5]/W4:10.4f} {H[0,t,9]/W4:10.4f} | {off}")

# ------------------------------------------------------------------ data
SEEDS = (1030, 1031)
acc = {}
parts = []
for seed in SEEDS:
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = prefix_stats(x)
    lev, levprev, T, clip = level_terms(x, H, Z)
    parts.append(dict(x=x, lev=lev, levprev=levprev, M=st["M1"], S=st["M2"], a=st["a1"], p1=st["p1"],
                      Mprev=st["M1prev"], aprev=st["a1prev"], h4=H[:, :, 4], h5=H[:, :, 5], h9=H[:, :, 9],
                      th=np.concatenate([np.zeros((len(x), 1)), theta4(H[:, :-1])], 1),
                      nclipC=(clip[:, :, CI] > 0).sum(-1), nclipW=(clip[:, :, WI] > 0).sum(-1),
                      **{"T_" + k: v for k, v in T.items()}))
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
x, lev, M, S, a = D["x"], D["lev"], D["M"], D["S"], D["a"]
Sp = pos(np.where(np.isfinite(S), S, 0))
n = len(x)
tt = np.broadcast_to(np.arange(10), x.shape)
TK = ["growth", "u4", "u5", "u9", "x", "clipC", "clipW"]
print(f"\n=== [MEASURED n = {n} sequences, N(0,1), seeds {SEEDS}] ===")

print("\n--- F5 re-check: level/(10.157 M) by age of the max at t (states with M > 0) ---")
print("age | n | median ratio | IQR | slope level/10.157 ~ M (no icpt) | R2c | slopes on (M, S+) | R2c")
for age in range(10):
    s = (a == age) & (M > 0)
    q = lev[s] / (W4 * M[s])
    b1, _, r1 = slope0(lev[s] / W4, [M[s]])
    b2, _, r2 = slope0(lev[s] / W4, [M[s], Sp[s]])
    print(f"{age} | {s.sum()} | {np.median(q):.4f} | [{np.percentile(q,25):.4f}, {np.percentile(q,75):.4f}] | {b1[0]:.4f} | {r1:.4f} | {b2[0]:.4f}, {b2[1]:.4f} | {r2:.4f}")

print("\n--- update terms by age of the max at t: least-squares slope through 0 of term/10.157 on M (M > 0). Sum of terms = level(t) - level(t-1) ---")
print("age | level(t) | level(t-1) | " + " | ".join(TK) + " | sum of terms")
for age in range(10):
    s = (a == age) & (M > 0) & (tt >= 1)
    den = (M[s] ** 2).sum() * W4
    sl = lambda y: (y[s] * M[s]).sum() / den
    terms = [sl(D["T_" + k]) for k in TK]
    print(f"{age} | {sl(lev):.4f} | {sl(D['levprev']):.4f} | " + " | ".join(f"{v:+.4f}" for v in terms) + f" | {sum(terms):+.4f}")

print("\n--- the same with two regressors: for age 0 and 1 use (M_new, M_old+) [M_old+ = positive part of the previous max], for age >= 2 use (M, S+) ---")
Mold = D["Mprev"]                      # at age 0: the max before the record (window x_0..x_{t-1})
Moldp = pos(np.where(np.isfinite(Mold), Mold, 0))
# for age 1 we need the max before the record at t-1: shift by one more step
Mold2 = np.concatenate([np.full((n, 1), 0.0), Moldp[:, :-1]], 1)
print("age | regressors | level(t) | level(t-1) | " + " | ".join(TK))
for age in range(10):
    s = (a == age) & (M > 0) & (tt >= 1)
    R2 = Moldp if age == 0 else (Mold2 if age == 1 else Sp)
    Xr = np.column_stack([M[s], R2[s]])
    G = np.linalg.inv(Xr.T @ Xr) @ Xr.T / W4
    for j, nm in enumerate(("M" if age >= 2 else "M_new", "S+" if age >= 2 else "M_old+")):
        print(f"{age} | {nm} | {(G @ lev[s])[j]:.4f} | {(G @ D['levprev'][s])[j]:.4f} | " + " | ".join(f"{(G @ D['T_' + k][s])[j]:+.4f}" for k in TK))

print("\n--- records: level one step later, level(s+1)/10.157 against (M_new, M_old+), by age of the OLD max at s-1; and the pieces lam^2*level(s-1), h4(s) ---")
print("age_old | n | level(s+1): b_new, b_old, R2c | lam^2 level(s-1): b_old | h4(s)/10.157 = x_s - theta4(s): b_new, b_old | other terms: b_new, b_old | theta4(s)/M_old slope")
rec = (a == 0) & (M > 0) & (tt >= 1) & (tt <= 8)
idx = np.argwhere(rec)
ii, ss = idx[:, 0], idx[:, 1]
keep = D["a"][ii, ss + 1] == 1                       # record still stands at s+1
ii, ss = ii[keep], ss[keep]
lev1 = D["lev"][ii, ss + 1] / W4
levm1 = D["levprev"][ii, ss] / W4
h4s = D["h4"][ii, ss] / W4
Mn, Mo, ao, ths = M[ii, ss], Moldp[ii, ss], D["aprev"][ii, ss], D["th"][ii, ss]
other = lev1 - lam ** 2 * levm1 - h4s
for age in list(range(8)) + ["first positive (M_old+ = 0)", "all with M_old > 0"]:
    if isinstance(age, int):
        s = (ao == age) & (Mo > 0)
    elif age.startswith("first"):
        s = Mo == 0
    else:
        s = Mo > 0
    if s.sum() < 300:
        continue
    if Mo[s].max() == 0:
        b, _, r = slope0(lev1[s], [Mn[s]])
        bo, _, _ = slope0(other[s], [Mn[s]])
        bh, _, _ = slope0(h4s[s], [Mn[s]])
        print(f"{age} | {s.sum()} | {b[0]:.4f}, -, {r:.5f} | - | {bh[0]:.4f}, - | {bo[0]:.4f}, - | -")
        continue
    Xr = [Mn[s], Mo[s]]
    b, _, r = slope0(lev1[s], Xr)
    bl, _, _ = slope0(lam ** 2 * levm1[s], [Mo[s]])
    bh, _, _ = slope0(h4s[s], Xr)
    bo, _, _ = slope0(other[s], Xr)
    bt, _, _ = slope0(ths[s], [Mo[s]])
    print(f"{age} | {s.sum()} | {b[0]:.4f}, {b[1]:.4f}, {r:.5f} | {bl[0]:.4f} | {bh[0]:.4f}, {bh[1]:.4f} | {bo[0]:.4f}, {bo[1]:.4f} | {bt[0]:.4f}")
q = lev1 / Mn
print(f"  ratio level(s+1)/(10.157 M_new): median {np.median(q):.4f}, IQR [{np.percentile(q,25):.4f}, {np.percentile(q,75):.4f}], mean {q.mean():.4f}; "
      f"first-positive only: median {np.median(q[Mo==0]):.4f}; M_old>0: median {np.median(q[Mo>0]):.4f}")

print("\n--- records: units 5 and 9 one step after the record (s+1) and their level cost at s+2, against (M_new, M_old+) ---")
keep2 = (ss + 2 <= 9)
i2, s2 = ii[keep2], ss[keep2]
keep3 = D["a"][i2, s2 + 2] == 2
i2, s2 = i2[keep3], s2[keep3]
Mn2, Mo2 = M[i2, s2], Moldp[i2, s2]
h5n, h9n, h4n = D["h5"][i2, s2 + 1] / W4, D["h9"][i2, s2 + 1] / W4, D["h4"][i2, s2 + 1] / W4
xn = x[i2, s2 + 1]
print("quantity | b_new | b_old | b_{x_{s+1}} | R2c   (regression through 0 on M_new, M_old+, x_{s+1})")
for nm, y in (("h4(s)/10.157", D["h4"][i2, s2] / W4), ("h5(s+1)/10.157", h5n), ("h9(s+1)/10.157", h9n), ("h4(s+1)/10.157 (false firing right after)", h4n),
              ("level(s+1)/10.157", D["lev"][i2, s2 + 1] / W4), ("level(s+2)/10.157", D["lev"][i2, s2 + 2] / W4),
              ("u5 term at s+2", D["T_u5"][i2, s2 + 2] / W4), ("u9 term at s+2", D["T_u9"][i2, s2 + 2] / W4),
              ("growth term at s+2", D["T_growth"][i2, s2 + 2] / W4), ("x term at s+2", D["T_x"][i2, s2 + 2] / W4),
              ("clipC at s+2", D["T_clipC"][i2, s2 + 2] / W4), ("clipW at s+2", D["T_clipW"][i2, s2 + 2] / W4)):
    b, _, r = slope0(y, [Mn2, Mo2, xn])
    print(f"{nm} | {b[0]:+.4f} | {b[1]:+.4f} | {b[2]:+.4f} | {r:.4f}")
print(f"  P(h5(s+1) > 0) = {(h5n > 0).mean():.4f}, P(h9(s+1) > 0) = {(h9n > 0).mean():.4f}  (n = {len(h5n)})")

print("\n--- drift at age >= 3 (max standing, pulses of the record gone): mean per-step terms / (10.157 M), split by what the input two steps / one step / zero steps back was ---")
print("(class of an input x_u relative to the stats before it: 'low' x_u <= S+, 'runner' S+ < x_u <= theta_4 (new runner-up, unit 4 silent), 'false' unit 4 fired without a record)")
th = D["th"]
Sprev = np.concatenate([np.zeros((n, 1)), Sp[:, :-1]], 1)                      # S+ of x_0..x_{u-1}
cls = np.where(x <= Sprev, 0, np.where(D["h4"] > 0, 2, 1))                      # for non-record steps
cls = np.where(x > pos(np.where(np.isfinite(D["Mprev"]), D["Mprev"], 0)), 3, cls)   # 3 = positive-part record
names = ["low", "runner", "false", "record"]
sel = (a >= 3) & (M > 0)
den = lambda s: (M[s] ** 2).sum() * W4
print("overall (age >= 3): " + ", ".join(f"{k} {((D['T_'+k][sel]*M[sel]).sum()/den(sel)):+.5f}" for k in TK)
      + f", total {sum((D['T_'+k][sel]*M[sel]).sum()/den(sel) for k in TK):+.5f}")
for lag, keys in ((0, ["x", "clipC", "clipW"]), (1, ["u4"]), (2, ["u5", "u9"])):
    c = np.concatenate([np.full((n, lag), -1), cls[:, : 10 - lag]], 1)
    print(f"  terms {keys} by class of x_(t-{lag}):")
    for ci in range(3):
        s = sel & (c == ci)
        share = s.sum() / sel.sum()
        print(f"    {names[ci]:6s}: share of steps {share:.4f}; " + ", ".join(f"{k} {((D['T_'+k][s]*M[s]).sum()/den(s)):+.5f}" for k in keys)
              + f";  contribution to the overall mean (M^2-weighted) " + ", ".join(f"{k} {((D['T_'+k][s]*M[s]).sum()/den(sel)):+.5f}" for k in keys))

print("\n--- drift of level/(10.157 M) with age for an isolated max (all other inputs <= 0) vs. all data: slope by age ---")
print("age | isolated: slope, n | all: slope | at least one positive input after the max: slope, n")
B = n
later_pos = np.zeros_like(x, bool)       # any positive input strictly after the max position, up to t
for t in range(10):
    cnt = np.zeros(B, int)
    for u in range(t + 1):
        cnt += ((u > D["p1"][:, t]) & (x[:, u] > 0))
    later_pos[:, t] = cnt > 0
for age in range(1, 10):
    s0 = (a == age) & (M > 0) & (Sp == 0)
    s1 = (a == age) & (M > 0)
    s2 = (a == age) & (M > 0) & later_pos
    f = lambda s: (lev[s] * M[s]).sum() / ((M[s] ** 2).sum() * W4) if s.sum() > 100 else float("nan")
    print(f"{age} | {f(s0):.4f}, {s0.sum()} | {f(s1):.4f} | {f(s2):.4f}, {s2.sum()}")
