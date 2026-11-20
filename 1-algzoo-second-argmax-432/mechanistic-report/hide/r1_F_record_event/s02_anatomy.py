"""s02 (task F1). Step-by-step anatomy of a record at step s (2 <= s <= 6 so that s+3 <= 9).

Record: x_s > M0 = max(x_0..x_{s-1}), M0 > 0.  g = s - p0 = age of the old max at step s.  S0 = old runner-up.
'Low' following inputs:  L1: x_{s+1..s+3} < M0 (= the runner-up S after the record)
                         L2: x_{s+1..s+3} < max(S0, 0)  (stricter: nothing enters the top three)
N(0,1), seeds 1001-1003, 400k sequences each.
"""
from fcommon import *

SEEDS = (1001, 1002, 1003)
rows = []           # one record per row
for seed in SEEDS:
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    for s in range(2, 7):
        rec, q = records(x, s, st)
        i = np.flatnonzero(rec)
        rows.append(dict(
            s=np.full(len(i), s), Mn=q["Mn"][i], M0=q["M0"][i], S0=q["S0"][i], S0p=q["S0p"][i], g=q["g"][i],
            xf=x[i, s + 1:s + 4],                       # following three inputs
            Hm=H[i, s - 1], H=H[i, s:s + 4], Z=Z[i, s:s + 4],
            M3=np.maximum(st["M3"][i, s - 1], 0) if s >= 3 else np.zeros(len(i))))
R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
n = len(R["s"])
G = R["Z"] > 0                                         # (n, 4 steps, 16)
L1 = (R["xf"] < R["M0"][:, None]).all(1)
L2 = (R["xf"] < R["S0p"][:, None]).all(1)
print(f"records (s=2..6, M0>0): n = {n};  L1 (next three inputs < M0): {L1.sum()} ({L1.mean():.3f});  "
      f"L2 (< max(S0,0)): {L2.sum()} ({L2.mean():.3f})")
print("distribution of g:", {int(g): int((R['g'] == g).sum()) for g in np.unique(R['g'])})

# ---------------------------------------------------------------- 1. gate frequencies
for name, sel in (("all records", np.ones(n, bool)), ("L1", L1), ("L2", L2)):
    print(f"\n=== P(unit on) around the record, {name}, n = {sel.sum()} ===")
    print("            unit " + " ".join(f"{u:6d}" for u in range(16)))
    print("    step s-1:    " + " ".join(f"{v:6.3f}" for v in (R["Hm"][sel] > 0).mean(0)))
    for d in range(4):
        print(f"    step s+{d}:    " + " ".join(f"{v:6.3f}" for v in G[sel, d].mean(0)))

comp_off_s = ~G[:, 0][:, COMP].any(1)
C0 = G[:, 0, 4] & comp_off_s
C1 = G[:, 1, 5] & G[:, 1, 9] & ~G[:, 1, 4] & G[:, 1][:, COMP].all(1) & G[:, 1, 0]
C2 = ~G[:, 2, 0] & ~G[:, 2][:, PULSE].any(1) & G[:, 2][:, COMP].all(1)
C3 = G[:, 3, 0] & ~G[:, 3][:, PULSE].any(1) & G[:, 3][:, COMP].all(1)
print("\n=== coverage of the pattern classes (fraction of records) ===")
print("  C_s   : unit 4 on, comparators 1,6,7,8 all off")
print("  C_s+1 : units 5, 9 on; unit 4 off; comparators on; unit 0 on")
print("  C_s+2 : unit 0 off; pulses 4,5,9 off; comparators on")
print("  C_s+3 : unit 0 on; pulses off; comparators on")
print(f"  {'':12s} {'all':>8s} {'L1':>8s} {'L2':>8s}")
for lab, c in (("unit 4 on @s", G[:, 0, 4]), ("C_s", C0), ("C_s+1", C1), ("C_s+2", C2), ("C_s+3", C3),
               ("all four", C0 & C1 & C2 & C3)):
    print(f"  {lab:12s} {c.mean():8.4f} {c[L1].mean():8.4f} {c[L2].mean():8.4f}")
print("  by s (L1): " + "  ".join(f"s={s}: {(C0 & C1 & C2 & C3)[L1 & (R['s'] == s)].mean():.3f}" for s in range(2, 7)))
print("  which comparators are still on at the record step s (all records): " +
      ", ".join(f"unit {u}: {G[:, 0, u].mean():.4f}" for u in COMP))
print("  wave units inside the classes (L1 & all four classes): P(on) at s, s+1, s+2, s+3")
cl = L1 & C0 & C1 & C2 & C3
for u in WAVE:
    print(f"    unit {u:2d}: " + " ".join(f"{G[cl, d, u].mean():6.3f}" for d in range(4)))
for d in range(4):
    pid = pattern_id(G[cl, d])
    vals, cnt = np.unique(pid, return_counts=True)
    o = np.argsort(-cnt)[:4]
    print(f"  step s+{d}: top full patterns within the classes: " +
          "; ".join(f"{pattern_str(int(vals[k]))} {cnt[k] / cl.sum():.3f}" for k in o))

# ---------------------------------------------------------------- 2. unit 4 at s
print("\n=== unit 4 at the record step ===")
Hm, H0, H1, H2, H3 = R["Hm"], R["H"][:, 0], R["H"][:, 1], R["H"][:, 2], R["H"][:, 3]
x_s, x1, x2, x3 = R["Mn"], R["xf"][:, 0], R["xf"][:, 1], R["xf"][:, 2]
th4 = -(Hm @ W[4]) / w[4]
err = np.abs(H0[:, 4] - w[4] * (x_s - th4))[G[:, 0, 4]].max()
print(f"[EXACT] h4(s) = w4 (x_s - theta4(s)),  theta4 = -W[4].h_(s-1)/w4;  max |error| on unit-4-on records: {err:.2e};"
      f"  unit 4 on in {G[:, 0, 4].mean():.5f} of records")
print("theta4(s) ~ a M0 + b S0+ (+ c M3+)  (no intercept), all records:")
b, r2 = ols(th4, [R["M0"]]); print(f"   M0 only         : {np.round(b, 4)}  R2 = {r2:.4f}")
b, r2 = ols(th4, [R["M0"], R["S0p"]]); print(f"   M0, S0+         : {np.round(b, 4)}  R2 = {r2:.4f}")
b, r2 = ols(th4, [R["M0"], R["S0p"], R["M3"]]); print(f"   M0, S0+, M3+    : {np.round(b, 4)}  R2 = {r2:.4f}")
print("   by g:   g   n      a(M0)   b(S0+)   R2     | median (M0 - theta4)/M0   | h4 ~ c1 Mn + c2 M0 + c3 S0+: c1, c2, c3, R2")
for g in range(1, 7):
    m = R["g"] == g
    b, r2 = ols(th4[m], [R["M0"][m], R["S0p"][m]])
    b4, r4 = ols(H0[m, 4], [R["Mn"][m], R["M0"][m], R["S0p"][m]])
    print(f"          {g}  {m.sum():6d}  {b[0]:6.3f}  {b[1]:6.3f}  {r2:.4f}  |  {np.median((R['M0'][m] - th4[m]) / R['M0'][m]):6.3f}"
          f"                  |  {b4[0]:7.3f} {b4[1]:7.3f} {b4[2]:7.3f}  {r4:.4f}")
b4, r4 = ols(H0[:, 4], [R["Mn"], R["M0"], R["S0p"]])
print(f"   pooled: h4(s) ~ {b4[0]:.3f} Mn + {b4[1]:.3f} M0 + {b4[2]:.3f} S0+   R2 = {r4:.4f}")
b4, r4 = ols(H0[:, 4], [R["Mn"] - R["M0"], R["M0"], R["S0p"]])
print(f"   pooled: h4(s) ~ {b4[0]:.3f} (Mn-M0) + {b4[1]:.3f} M0 + {b4[2]:.3f} S0+   R2 = {r4:.4f}")
b4, r4 = ols(H0[:, 4], [R["Mn"] - R["M0"]])
print(f"   pooled: h4(s) ~ {b4[0]:.3f} (Mn-M0)   R2 = {r4:.4f};   mean h4 = {H0[:, 4].mean():.2f}, mean 10.157(Mn-M0) = {(w[4] * (R['Mn'] - R['M0'])).mean():.2f}")

# ---------------------------------------------------------------- 3. units 5, 9 at s+1
print("\n=== units 5 and 9 one step after the record ===")
NC = [u for u in range(16) if u not in COMP]            # units that can be non-zero at s on class C_s
for pu in (5, 9):
    pred = w[pu] * x1 + H0[:, NC] @ W[pu, NC]
    on = C0 & G[:, 1, pu]
    print(f"[EXACT] h{pu}(s+1) = w{pu} x_(s+1) + sum_(u not comparator) W[{pu},u] h_u(s)  on C_s & unit {pu} on at s+1 "
          f"(coverage all {on.mean():.4f}, L1 {on[L1].mean():.4f}, L2 {on[L2].mean():.4f}); max |error| = {np.abs(H1[on, pu] - pred[on]).max():.2e}")
    print(f"        = {w[pu]:.3f} x_(s+1) " + " ".join(f"{W[pu, u]:+.3f} h{u}" for u in NC))
    # size of the terms
    terms = {"w x_(s+1)": w[pu] * x1, "W[.,4] h4(s)": W[pu, 4] * H0[:, 4],
             "wave units (s)": H0[:, WAVE] @ W[pu, WAVE], "pulses 5,9 (s)": H0[:, [5, 9]] @ W[pu, [5, 9]]}
    print("        mean (sd) of the terms on L1 records: " + "; ".join(f"{k}: {v[L1].mean():.2f} ({v[L1].std():.2f})" for k, v in terms.items())
          + f"; total h{pu}(s+1): {H1[L1, pu].mean():.2f} ({H1[L1, pu].std():.2f})")
    for name, sel in (("all", np.ones(n, bool)), ("L1", L1), ("L2", L2)):
        y = H1[sel, pu]
        cols = [R["Mn"][sel] - R["M0"][sel], R["M0"][sel], R["S0p"][sel]]
        b, r2 = ols(y, cols)
        b2, r22 = ols(y, cols + [x1[sel]])
        print(f"   [FIT] {name:3s}: h{pu}(s+1) ~ {b[0]:.3f} (Mn-M0) + {b[1]:.3f} M0 + {b[2]:.3f} S0+   R2 = {r2:.4f}   "
              f"| with x_(s+1): {np.round(b2, 3)}  R2 = {r22:.4f}")
    print("   by g (L1):  g    n     (Mn-M0)    M0     S0+     R2    | + x_(s+1) coef, R2")
    for g in range(1, 7):
        m = L1 & (R["g"] == g)
        cols = [R["Mn"][m] - R["M0"][m], R["M0"][m], R["S0p"][m]]
        b, r2 = ols(H1[m, pu], cols)
        b2, r22 = ols(H1[m, pu], cols + [x1[m]])
        print(f"               {g} {m.sum():6d}  {b[0]:7.3f} {b[1]:7.3f} {b[2]:7.3f}  {r2:.4f}  |  {b2[3]:7.3f}  {r22:.4f}")

# ---------------------------------------------------------------- 4. unit 0
print("\n=== unit 0 at s+1, s+2, s+3 ===")
Z0 = R["Z"][:, :, 0]
for name, sel in (("all", np.ones(n, bool)), ("L1", L1), ("L2", L2)):
    print(f"  {name:3s}: P(unit 0 on) at s, s+1, s+2, s+3 = " + " ".join(f"{(Z0[sel, d] > 0).mean():.4f}" for d in range(4))
          + "   median pre-activation: " + " ".join(f"{np.median(Z0[sel, d]):7.2f}" for d in range(4)))
pred = w[0] * x2 + H1 @ W[0]
print(f"[EXACT] z0(s+2) = w0 x_(s+2) + W[0].h_(s+1): max |error| = {np.abs(Z0[:, 2] - pred).max():.2e}")
terms = {"-0.913 h5(s+1)": W[0, 5] * H1[:, 5], "-1.424 h9(s+1)": W[0, 9] * H1[:, 9], "0.595 h4(s+1)": W[0, 4] * H1[:, 4],
         "comparators(s+1)": H1[:, COMP] @ W[0, COMP], "wave(s+1)": H1[:, WAVE] @ W[0, WAVE], "w0 x_(s+2)": w[0] * x2}
print("  terms of z0(s+2), mean (sd) on L1: " + "; ".join(f"{k}: {v[L1].mean():.2f} ({v[L1].std():.2f})" for k, v in terms.items())
      + f"; total {Z0[L1, 2].mean():.2f} ({Z0[L1, 2].std():.2f})")
print("  for comparison the same terms of z0(s+3) from h_(s+2), mean on L1: " + "; ".join(
    f"{k}: {v[L1].mean():.2f}" for k, v in {"h5": W[0, 5] * H2[:, 5], "h9": W[0, 9] * H2[:, 9], "comparators": H2[:, COMP] @ W[0, COMP],
                                            "wave": H2[:, WAVE] @ W[0, WAVE]}.items()) + f"; total {Z0[L1, 3].mean():.2f}")
print("  x-coefficient of the comparator term: sum_j W[0,j] w_j =", W[0, COMP] @ w[COMP], " (the input cancels: unit 0 reads threshold differences)")
for name, sel in (("L1", L1), ("L2", L2)):
    cols = [R["Mn"][sel] - R["M0"][sel], R["M0"][sel], R["S0p"][sel]]
    for d in (1, 2, 3):
        b, r2 = ols(Z0[sel, d], cols)
        print(f"   [FIT] {name}: z0(s+{d}) ~ {b[0]:7.3f} (Mn-M0) + {b[1]:7.3f} M0 + {b[2]:7.3f} S0+   R2 = {r2:.4f}")
# what would unit 0 have been at s+2 without the two pulses?
z_nop = Z0[:, 2] - W[0, 5] * H1[:, 5] - W[0, 9] * H1[:, 9]
print(f"  without the terms of units 5 and 9, z0(s+2) would be > 0 in {(z_nop[L1] > 0).mean():.4f} of L1 records (median {np.median(z_nop[L1]):.2f});"
      f" without unit 9 only: {((Z0[:, 2] - W[0, 9] * H1[:, 9])[L1] > 0).mean():.4f}; without unit 5 only: {((Z0[:, 2] - W[0, 5] * H1[:, 5])[L1] > 0).mean():.4f}")

# ---------------------------------------------------------------- 5. comparators and thresholds after the record
print("\n=== comparator thresholds after the record: theta_j(s+d) ~ a Mn + b M0 + c S0+  (L2 records, no intercept) ===")
states = [Hm, H0, H1, H2]
for j in COMP + [4]:
    for d in range(0, 4):
        th = -(states[d] @ W[j]) / w[j]
        b, r2 = ols(th[L2], [R["Mn"][L2], R["M0"][L2], R["S0p"][L2]])
        print(f"   theta_{j}(s+{d}) ~ {b[0]:6.3f} Mn + {b[1]:6.3f} M0 + {b[2]:6.3f} S0+   R2 = {r2:.4f}")

np.savez(HIDE / "r1_F_record_event" / "s02_fits.npz", dummy=0)
