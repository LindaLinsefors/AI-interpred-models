"""C2. Restart of the clock at a record, from the weights and on data (400k N(0,1) sequences, seed 1200)."""
from cl import *

# =============================================================== A. exact quantities from the weights
print("=== A. [EXACT] from the weights ===")
print("LEFT['clock'] on the wave units (|.|@deg):  " + "  ".join(f"u{u}:{fmt(LC[ORD.index(u)])}" for u in WAVE))
print("coordinates of the pulse columns:  " + "  ".join(f"LC.W[ORD,{p}] = {fmt(CP[p])}" for p in PULSE) + f"   LC.w[ORD] = {fmt(ECHO)}")
print("\nSurvival through 'all four comparators zero':  z_after = sum_n s_n m_n + (for pairs) s~_n conj(m_n),")
print("  s_n = LC[wave].RIGHT_n[wave],  s~_n = LC[wave].conj(RIGHT_n[wave])")
for n in NAMES:
    s = LC[wi] @ RIGHT[n][wi]
    line = f"  {n:7s}: s = {fmt(s, 4)}"
    if IS_COMPLEX[n]:
        line += f"   s~ = {fmt(LC[wi] @ np.conj(RIGHT[n][wi]), 4)}"
    print(line)
print("  (check: s_clock + LC[comp].RC[comp] =", np.round(LC[wi] @ RC[wi] + LC[ci] @ RC[ci], 6), ")")

# impulse: exact two-step account
for bg in (0.0, -1.0):
    x = np.full((1, 10), bg); x[0, 0] = 1.0
    H, Z = run(x, return_pre=True)
    tm = clock_terms(x, H, Z)
    zz = tm["z"][0] / K4
    print(f"\nimpulse x = (1, {bg}, {bg}, ...): h4(0)/K4 = {H[0, 0, 4] / K4:.3f};  z(t)/K4 for t = 0..9:")
    print("   " + "  ".join(fmt(v) for v in zz))
    print("   phase increments:", np.round(wrap(np.diff(deg(zz[1:]))), 1), " modulus ratios:", np.round(abs(zz[2:]) / abs(zz[1:-1]), 3))
    for t in (1, 2, 3):
        parts = {k: tm[k][0, t - 1] / K4 for k in tm if k not in ("z", "check")}
        print(f"   t={t}: " + "  ".join(f"{k} {fmt(v)}" for k, v in parts.items() if abs(v) > 1e-4)
              + f"   | h5(t-1) = {H[0, t - 1, 5]:.2f}, h9(t-1) = {H[0, t - 1, 9]:.2f}")
print("\nprediction for a pure impulse (h5(s+1) = W[5,4] h4, h9(s+1) = W[9,4] h4, no clipping):")
z2 = LAMC + CP[5] * W[5, 4] + CP[9] * W[9, 4]
print(f"   z(s+2)/z(s+1) = lam + CP5 W[5,4] + CP9 W[9,4] = {fmt(LAMC)} + {fmt(CP[5] * W[5, 4])} + {fmt(CP[9] * W[9, 4])} = {fmt(z2)}")

# =============================================================== B. data
x, H, Z, st = gen(400_000, 1200)
tm = clock_terms(x, H, Z)
print("\n=== B. [MEASURED n=400k sequences, seed 1200] ===   decomposition check (max abs error):", tm["check"])
z = tm["z"] / K4
M, S, Tq, a1, a2 = st["M1"], pos(st["M2"]), pos(st["M3"]), st["a1"], st["a2"]
G = (Z > 0)
th = thresholds(H)                                   # (B, T, 5): thresholds for step t+1, order 1,6,7,8,4

# record steps s = 1..8 over a positive old max
B_, s_ = np.nonzero((a1[:, 1:9] == 0) & (M[:, 0:8] > 0)); s_ = s_ + 1
print(f"records at s=1..8 over a positive old max: {B_.size}")
compoff = ~G[B_, s_][:, COMP]
print("  comparators off at the record step (1,6,7,8):", np.round(compoff.mean(0), 4), "  all four off:", np.round(compoff.all(1).mean(), 4))
print("  unit 4 on at the record step:", np.round(G[B_, s_, 4].mean(), 5), "  units 5, 9 on at the record step:", np.round(G[B_, s_][:, [5, 9]].mean(0), 4))
print("  wave units on at the record step " + str(WAVE) + ":", np.round(G[B_, s_][:, WAVE].mean(0), 3))
nonrec = (a1[:, 1:] > 0)
print("  unit 4 on at non-record steps t>=1:", np.round(G[:, 1:, 4][nonrec].mean(), 5))

Mold, Sold, aold = M[B_, s_ - 1], S[B_, s_ - 1], a1[B_, s_ - 1]
Mnew = M[B_, s_]
h4 = H[B_, s_, 4] / K4
th4 = th[B_, s_ - 1, 4]
print("\n  unit-4 kick h4(s)/K4 against the true gap M_new - M_old:")
print(f"    regression h4/K4 ~ a (Mnew - Mold): a = {np.sum(h4 * (Mnew - Mold)) / np.sum((Mnew - Mold) ** 2):.4f}, "
      f"R2 = {1 - np.sum((h4 - (Mnew - Mold)) ** 2) / np.sum(h4 ** 2):.4f} (for a = 1)")
print("    (M_old - theta_4)/(M_old - S_old+) by age of the old max at s-1 [median, IQR], and share of records with theta_4 > M_new (missed)")
for a in range(0, 9):
    q = (aold == a)
    if q.sum() < 300: continue
    sag = (Mold[q] - th4[q]) / (Mold[q] - Sold[q])
    print(f"      a_old={a}: {np.median(sag):+.3f} [{np.percentile(sag, 25):+.3f}, {np.percentile(sag, 75):+.3f}]   missed {np.mean(h4[q] <= 0):.4f}   n={q.sum()}")

# z at the record step
zs = z[B_, s_]
zprev = z[B_, s_ - 1]
print("\n  z(s)/K4 at the record step (comparators zero -> wave units only):")
print(f"    rms |z(s)| = {np.sqrt(np.mean(abs(zs) ** 2)):.4f}   rms |z(s-1)| = {np.sqrt(np.mean(abs(zprev) ** 2)):.4f}   rms h4/K4 = {np.sqrt(np.mean(h4 ** 2)):.4f}")
c, e = cfit(zs, [Mnew, Mold, Sold]); print(f"    z(s) ~ a Mnew + b Mold + c Sold:  a = {fmt(c[0])}  b = {fmt(c[1])}  c = {fmt(c[2])}  EV = {e:.3f}")
c, e = cfit(zs, [Mnew, Mold, Sold, zprev, np.conj(zprev)])
print(f"    z(s) ~ a Mnew + b Mold + c Sold + d z(s-1) + e conj z(s-1):  a = {fmt(c[0])}  b = {fmt(c[1])}  c = {fmt(c[2])}  d = {fmt(c[3])}  e = {fmt(c[4])}  EV = {e:.3f}")
print(f"    exact prediction for d, e (from A): lam * s_clock = {fmt(LAMC * (LC[wi] @ RC[wi]))},  conj(lam) * s~_clock = {fmt(np.conj(LAMC) * (LC[wi] @ np.conj(RC[wi])))}")

# ---- step s+1: exact terms
print("\n  z(s+1) = h4(s) + lam z(s) + CP5 h5(s) + CP9 h9(s) + ECHO x_{s+1} + clips(s+1); all terms /K4.")
names = ["p4", "carried", "p5", "p9", "echo", "clip1", "clip6", "clip7", "clip8", "clipwave"]
for label, cond in (("all records", np.ones(B_.size, bool)),
                    ("quiet next step (x_{s+1} < M_old, i.e. no new max / runner-up at s+1; S+ unchanged)", (a1[B_, s_ + 1] == 1) & (a2[B_, s_ + 1] >= 2)),
                    ("next step x_{s+1} between M_old and M_new (a2 = 0 at s+1)", (a1[B_, s_ + 1] == 1) & (a2[B_, s_ + 1] == 0))):
    print(f"  --- {label}: n = {cond.sum()}")
    z1 = z[B_, s_ + 1][cond]
    gap = (Mnew - Mold)[cond]
    print(f"      z(s+1) ~ c (Mnew - Mold): c = {fmt(cfit(z1, [gap])[0][0])}, EV = {cfit(z1, [gap])[1]:.4f};   z(s+1) ~ c h4(s)/K4: c = {fmt(cfit(z1, [h4[cond]])[0][0])}, EV = {cfit(z1, [h4[cond]])[1]:.4f}")
    print("      term      rms/rms(h4)   LS coefficient on h4(s)/K4   share of gate on")
    for k in names:
        v = tm[k][B_, s_][cond] / K4                                         # index s_ <-> t = s_+1
        cc = cfit(v, [h4[cond]])[0][0]
        print(f"      {k:9s}  {np.sqrt(np.mean(abs(v) ** 2)) / np.sqrt(np.mean(h4[cond] ** 2)):8.4f}      {fmt(cc, 4)}        {np.mean(abs(v) > 0):.3f}")
    r = z1 - h4[cond]
    print(f"      remainder r = z(s+1)/K4 - h4(s)/K4: rms {np.sqrt(np.mean(abs(r) ** 2)):.4f} vs rms h4/K4 {np.sqrt(np.mean(h4[cond] ** 2)):.4f}  (sum|r|^2/sum|h4|^2 = {np.sum(abs(r) ** 2) / np.sum(h4[cond] ** 2):.4f})")
    zp, Mo, So, Mn = zprev[cond], Mold[cond], Sold[cond], Mnew[cond]
    for lab, cols in (("z(s-1)", [zp]), ("z(s-1), conj", [zp, np.conj(zp)]), ("Mold, Sold", [Mo, So]), ("z(s-1), conj, Mold, Sold", [zp, np.conj(zp), Mo, So]),
                      ("z(s-1), conj, Mold, Sold, Mnew, x_{s+1}", [zp, np.conj(zp), Mo, So, Mn, x[B_, s_ + 1][cond]])):
        c, e = cfit(r, cols)
        print(f"      r ~ {lab:42s}: EV {e:.3f}  coefs " + "  ".join(fmt(v) for v in c))
    # by age of the old max
    print("      by age of the old max at s-1:  r ~ b Mold + c Sold  (the old clock is ~ (Mold - Sold) rho e^{i phi(a_old)})")
    for a in range(0, 8):
        q = aold[cond] == a
        if q.sum() < 1000: continue
        c, e = cfit(r[q], [Mo[q], So[q]])
        cz, ez = cfit(r[q], [zp[q], np.conj(zp[q])])
        print(f"        a_old={a}: b = {fmt(c[0])}  c = {fmt(c[1])}  EV {e:.3f} | r ~ d z(s-1) + e conj: d = {fmt(cz[0])} e = {fmt(cz[1])} EV {ez:.3f} | rms r {np.sqrt(np.mean(abs(r[q]) ** 2)):.3f}  rms z(s-1) {np.sqrt(np.mean(abs(zp[q]) ** 2)):.3f}  n={q.sum()}")

# ---- step s+2: why 37 deg
print("\n  step s+2 (a1 = 2):  z(s+2) = lam z(s+1) + CP4 h4(s+1) + CP5 h5(s+1) + CP9 h9(s+1) + ECHO x_{s+2} + clips(s+2)")
B2, s2 = np.nonzero((a1[:, 1:8] == 0) & (M[:, 0:7] > 0)); s2 = s2 + 1
for label, cond in (("all with a1 = 2 at s+2", a1[B2, s2 + 2] == 2),
                    ("quiet: no new runner-up at s+1 or s+2 (a2 >= 3 at s+2)", (a1[B2, s2 + 2] == 2) & (a2[B2, s2 + 2] >= 3))):
    Bq, sq = B2[cond], s2[cond]
    z1, zz2 = z[Bq, sq + 1], z[Bq, sq + 2]
    h4s = H[Bq, sq, 4] / K4
    print(f"  --- {label}: n = {cond.sum()}")
    print(f"      activations at s+1 relative to h4(s): h5/h4 median {np.median(H[Bq, sq + 1, 5] / K4 / h4s):.3f} [IQR {np.percentile(H[Bq, sq + 1, 5] / K4 / h4s, 25):.3f}, {np.percentile(H[Bq, sq + 1, 5] / K4 / h4s, 75):.3f}],"
          f" on {np.mean(H[Bq, sq + 1, 5] > 0):.3f};  h9/h4 median {np.median(H[Bq, sq + 1, 9] / K4 / h4s):.3f} [IQR {np.percentile(H[Bq, sq + 1, 9] / K4 / h4s, 25):.3f}, {np.percentile(H[Bq, sq + 1, 9] / K4 / h4s, 75):.3f}], on {np.mean(H[Bq, sq + 1, 9] > 0):.3f}"
          f"   (W[5,4] = {W[5, 4]:.3f}, W[9,4] = {W[9, 4]:.3f});  h4(s+1) on: {np.mean(H[Bq, sq + 1, 4] > 0):.4f}")
    print("      term      LS coefficient on z(s+1)   (sum = z(s+2)/z(s+1) in the LS sense)")
    tot = 0
    for k in names:
        v = tm[k][Bq, sq + 1] / K4
        cc = cfit(v, [z1])[0][0]; tot += cc
        print(f"      {k:9s}  {fmt(cc, 4)}    rms/rms z(s+1) {np.sqrt(np.mean(abs(v) ** 2)) / np.sqrt(np.mean(abs(z1) ** 2)):.4f}")
    print(f"      sum       {fmt(tot, 4)}    direct LS z(s+2) ~ c z(s+1): {fmt(cfit(zz2, [z1])[0][0], 4)}  EV {cfit(zz2, [z1])[1]:.4f}")
    d = wrap(deg(zz2) - deg(z1))
    print(f"      phase advance arg z(s+2) - arg z(s+1): median {np.median(d):.1f}, IQR [{np.percentile(d, 25):.1f}, {np.percentile(d, 75):.1f}]")
