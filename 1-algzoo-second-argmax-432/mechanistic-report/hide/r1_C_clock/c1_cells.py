"""C1. Test F6: per cell (a1, a2) fit  z/10.157 ~ alpha (M - S+) + beta S+ + gamma T+   (complex LS, no intercept).

z = clock coordinate of h_t; M, S, T = three largest of x_0..x_t; S+, T+ = positive parts; a1, a2 = ages of max / runner-up.
States with M <= 0 are left out of the fits (reported separately).   1.25M sequences, seeds 1100..1104.
"""
from cl import *

NB, BS, SEED0 = 5, 250_000, 1100
zs, Ms, Ss, Ts, A1, A2, A3, TT, XT = ([] for _ in range(9))
for b in range(NB):
    x, H, Z, st = gen(BS, SEED0 + b)
    zs.append((H[..., ORD] @ LC / K4).ravel())
    Ms.append(st["M1"].ravel()); Ss.append(pos(st["M2"]).ravel()); Ts.append(pos(st["M3"]).ravel())
    A1.append(st["a1"].ravel().astype(np.int8)); A2.append(st["a2"].ravel().astype(np.int8)); A3.append(st["a3"].ravel().astype(np.int8))
    TT.append(np.broadcast_to(np.arange(T), x.shape).ravel().astype(np.int8)); XT.append(x.ravel())
    del H, Z, st
z, M, S, Tq, a1, a2, a3, tt, xt = (np.concatenate(v) for v in (zs, Ms, Ss, Ts, A1, A2, A3, TT, XT))
del zs, Ms, Ss, Ts, A1, A2, A3, TT, XT
ok = M > 0
G = M - S
print(f"sequences {NB * BS}, states {z.size}, states with M > 0: {ok.mean():.4f}")
print(f"mean |z|^2 (z/10.157): M>0 states {np.mean(abs(z[ok])**2):.4f};  M<=0 states {np.mean(abs(z[~ok])**2):.5f}"
      f"  (share of total sum|z|^2 from M<=0 states: {np.sum(abs(z[~ok])**2) / np.sum(abs(z)**2):.5f})")

# ---------------------------------------------------------------- per-cell fits
cells = {}
for A in range(10):
    for B in range(-1, 10):
        s = ok & (a1 == A) & (a2 == B)
        n = int(s.sum())
        if n < 2000:
            continue
        y = z[s]
        c3, e3 = cfit(y, [G[s], S[s], Tq[s]])
        c2, e2 = cfit(y, [M[s], S[s]])
        cg, eg = cfit(y, [G[s]])
        cm, em = cfit(y, [M[s]])
        cells[(A, B)] = dict(n=n, c3=c3, e3=e3, c2=c2, e2=e2, cg=cg, eg=eg, cm=cm, em=em, ss=float((abs(y) ** 2).sum()))

def table(title, f, width=15):
    print("\n" + title)
    print("a1\\a2 " + "".join(f"{B:>{width}d}" for B in range(-1, 10)))
    for A in range(10):
        print(f"  {A}   " + "".join(f"{(f(cells[(A, B)]) if (A, B) in cells else '.'):>{width}s}" for B in range(-1, 10)))

table("n per cell (M > 0)", lambda c: str(c["n"]), 9)
table("alpha(a1, a2) of the 3-term fit:  |alpha|@phase(deg)", lambda c: fmt(c["c3"][0]))
table("beta(a1, a2):  |beta|@phase", lambda c: fmt(c["c3"][1]))
table("gamma(a1, a2):  |gamma|@phase", lambda c: fmt(c["c3"][2]))
table("explained fraction of sum|z|^2, 3-term fit", lambda c: f"{c['e3']:.4f}", 9)
table("explained fraction, alpha (M - S+) only", lambda c: f"{c['eg']:.4f}", 9)
table("explained fraction, alpha M + beta S+ (free)", lambda c: f"{c['e2']:.4f}", 9)
table("explained fraction, alpha M only", lambda c: f"{c['em']:.4f}", 9)
table("free fit alpha M + beta S+:  beta/alpha as |.|@phase   (-1 = 1@180 means exactly M - S)", lambda c: fmt(c["c2"][1] / c["c2"][0]))

# ---------------------------------------------------------------- is alpha a function of a1 only?
print("\nalpha by a1: spread over a2 cells (3-term fit).  'old' = runner-up older than max (a2 > a1), 'young' = a2 < a1")
print(" a1 | pooled-over-a2 alpha (G-only fit) | cells a2>a1: |alpha| range, phase range | cells 1<=a2<a1: |alpha| range, phase range | a2=0 cell alpha")
phi = {}
alpha1 = {}
for A in range(10):
    s = ok & (a1 == A)
    c, e = cfit(z[s], [G[s]])
    alpha1[A] = c[0]; phi[A] = np.angle(c[0])
    def rng(Bs):
        al = np.array([cells[(A, B)]["c3"][0] for B in Bs if (A, B) in cells])
        if al.size == 0:
            return "      -      "
        ph = wrap(deg(al) - deg(c[0]))
        return f"{abs(al).min():.3f}-{abs(al).max():.3f}, dphase {ph.min():+5.1f}..{ph.max():+5.1f}"
    a0 = fmt(cells[(A, 0)]["c3"][0]) if (A, 0) in cells else "-"
    print(f"  {A} | {fmt(c[0])}  EV {e:.4f} | {rng(range(A + 1, 10))} | {rng(range(1, A))} | {a0}")

# ---------------------------------------------------------------- pooled models, by a1 and overall
print("\nPooled comparison of parametrisations (coefficients depend on a1 only / on the cell (a1,a2)); explained fraction of sum|z|^2")
print(" a1 |    n     | a1-only coefficients: G only   M,S free   M only   G,S,T   G,S,T,x_t | per-cell coefficients: G only   M,S free   M only   G,S,T")
tot = np.zeros(10); totss = 0.0
for A in range(10):
    s = ok & (a1 == A)
    y = z[s]; ss = (abs(y) ** 2).sum()
    ev = [cfit(y, cols)[1] for cols in ([G[s]], [M[s], S[s]], [M[s]], [G[s], S[s], Tq[s]], [G[s], S[s], Tq[s], xt[s]])]
    pc = []
    for key in ("eg", "e2", "em", "e3"):
        res = sum((1 - c[key]) * c["ss"] for (AA, B), c in cells.items() if AA == A)
        covered = sum(c["ss"] for (AA, B), c in cells.items() if AA == A)
        pc.append(1 - res / covered)
    print(f"  {A} | {int(s.sum()):8d} |   {ev[0]:.4f}   {ev[1]:.4f}   {ev[2]:.4f}   {ev[3]:.4f}   {ev[4]:.4f}   |   {pc[0]:.4f}   {pc[1]:.4f}   {pc[2]:.4f}   {pc[3]:.4f}")
    if A >= 1:
        tot[:5] += (1 - np.array(ev)) * ss; tot[5:9] += (1 - np.array(pc)) * ss; totss += ss
print(f" a1>=1 pooled: a1-only: G {1 - tot[0]/totss:.4f}  M,S {1 - tot[1]/totss:.4f}  M {1 - tot[2]/totss:.4f}  G,S,T {1 - tot[3]/totss:.4f}  G,S,T,x {1 - tot[4]/totss:.4f}"
      f" | per-cell: G {1 - tot[5]/totss:.4f}  M,S {1 - tot[6]/totss:.4f}  M {1 - tot[7]/totss:.4f}  G,S,T {1 - tot[8]/totss:.4f}")

print("\nFree fit z ~ alpha M + beta S+ with a1-only coefficients: alpha, beta, beta/alpha")
for A in range(10):
    s = ok & (a1 == A)
    c, e = cfit(z[s], [M[s], S[s]])
    print(f"  a1={A}: alpha {fmt(c[0])}  beta {fmt(c[1])}  beta/alpha {fmt(c[1] / c[0])}")

# ---------------------------------------------------------------- radius: |z| against M - S, M, (M, S)
print("\nRadius only: |z|/10.157 fitted by real LS without intercept, by a1 (R2 = 1 - SSres/SS|z|^2), and Pearson r")
print(" a1 | rho in |z| ~ rho (M-S+): rho, R2, r | |z| ~ a M + b S+: a, b, b/a, R2 | |z| ~ a M: a, R2, r")
for A in range(1, 10):
    s = ok & (a1 == A)
    r = abs(z[s])
    def rfit(cols):
        X = np.column_stack(cols); c, *_ = np.linalg.lstsq(X, r, rcond=None)
        return c, 1 - ((r - X @ c) ** 2).sum() / (r ** 2).sum()
    cg_, eg_ = rfit([G[s]]); c2_, e2_ = rfit([M[s], S[s]]); cm_, em_ = rfit([M[s]])
    print(f"  {A} | {cg_[0]:.3f}  {eg_:.4f}  {np.corrcoef(r, G[s])[0, 1]:.3f} | {c2_[0]:.3f} {c2_[1]:+.3f}  {c2_[1] / c2_[0]:+.3f}  {e2_:.4f} | {cm_[0]:.3f}  {em_:.4f}  {np.corrcoef(r, M[s])[0, 1]:.3f}")

# ---------------------------------------------------------------- phase error around phi(a1)
print("\nphi(a1) = phase of the pooled alpha(a1) (G-only fit), deg:", {A: round(float(np.degrees(phi[A])) % 360, 1) for A in range(1, 10)})
print("rho(a1) = |alpha(a1)|:", {A: round(float(abs(alpha1[A])), 3) for A in range(1, 10)})
print("increments phi(a+1) - phi(a):", [round(float(wrap(np.degrees(phi[A + 1] - phi[A]))), 1) for A in range(1, 9)])
qs = [5, 25, 50, 75, 95]
ratio = G / np.where(ok, M, 1.0)
rb = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0000001]
print("\nphase error = arg(z) - phi(a1) in deg: percentiles 5/25/50/75/95, and share with |error| < 21.4 deg (half a clock step)")
for A in range(1, 10):
    s = ok & (a1 == A)
    err = wrap(deg(z[s]) - np.degrees(phi[A]))
    p = np.percentile(err, qs)
    print(f"  a1={A}: " + " ".join(f"{v:+7.1f}" for v in p) + f"   within 21.4: {np.mean(abs(err) < 21.37):.4f}   mean|err| {np.mean(abs(err)):.1f}   n={int(s.sum())}")
print("\nby ratio (M - S+)/M, all a1 >= 1 pooled:")
sel = ok & (a1 >= 1)
phi_arr = np.array([phi[A] for A in range(10)])
err_all = wrap(deg(z) - np.degrees(phi_arr[a1]))
for lo, hi in zip(rb[:-1], rb[1:]):
    s = sel & (ratio >= lo) & (ratio < hi)
    p = np.percentile(err_all[s], qs)
    print(f"  ratio [{lo:.2f},{min(hi, 1):.2f}): share of states {s.sum() / sel.sum():.3f}  " + " ".join(f"{v:+7.1f}" for v in p)
          + f"   within 21.4: {np.mean(abs(err_all[s]) < 21.37):.4f}   mean|err| {np.mean(abs(err_all[s])):.1f}")
print("\nshare with |phase error| < 21.4 deg, a1 (rows) x ratio bin (columns)")
print("        " + "".join(f"[{lo:.2f},{min(hi, 1):.2f})  " for lo, hi in zip(rb[:-1], rb[1:])))
for A in range(1, 10):
    row = []
    for lo, hi in zip(rb[:-1], rb[1:]):
        s = ok & (a1 == A) & (ratio >= lo) & (ratio < hi)
        row.append(f"{np.mean(abs(err_all[s]) < 21.37):.3f}" if s.sum() > 500 else "  -  ")
    print(f"  a1={A}: " + "       ".join(row))
print("\nmedian phase error (deg), a1 x ratio bin")
for A in range(1, 10):
    row = []
    for lo, hi in zip(rb[:-1], rb[1:]):
        s = ok & (a1 == A) & (ratio >= lo) & (ratio < hi)
        row.append(f"{np.median(err_all[s]):+6.1f}" if s.sum() > 500 else "   -  ")
    print(f"  a1={A}: " + "      ".join(row))


# ---------------------------------------------------------------- 3-term fit with a1-only coefficients; reference phase phi3(a1)
print("\n3-term fit with coefficients depending on a1 only: alpha, beta, gamma, EV; and split-half stability (seeds 1100-1101 vs 1102-1104)")
half = np.arange(z.size) < 2 * BS * T
phi3 = {}; alpha3 = {}
for A in range(10):
    s = ok & (a1 == A)
    c, e = cfit(z[s], [G[s], S[s], Tq[s]])
    ca, _ = cfit(z[s & half], [G[s & half], S[s & half], Tq[s & half]])
    cb, _ = cfit(z[s & ~half], [G[s & ~half], S[s & ~half], Tq[s & ~half]])
    phi3[A] = np.angle(c[0]); alpha3[A] = c[0]
    print(f"  a1={A}: alpha {fmt(c[0])}  beta {fmt(c[1])}  gamma {fmt(c[2])}  EV {e:.4f}   max|coef diff between halves| {abs(ca - cb).max():.4f}")
print("phi3(a1) deg:", {A: round(float(np.degrees(phi3[A])) % 360, 1) for A in range(1, 10)})
print("increments:", [round(float(wrap(np.degrees(phi3[A + 1] - phi3[A]))), 1) for A in range(1, 9)])
print("free-rotation prediction phi3(2) + (a-2)*42.75:", [round(float(np.degrees(phi3[2]) + (A - 2) * np.degrees(OMEGA)) % 360, 1) for A in range(2, 10)])
print("|alpha3(a+1)|/|alpha3(a)|:", [round(float(abs(alpha3[A + 1]) / abs(alpha3[A])), 3) for A in range(1, 9)], " |lam_clock| =", round(abs(LAMC), 4))

# split-half stability of the per-cell coefficients
dmax = 0
for (A, B), c in cells.items():
    if A == 0: continue
    s = ok & (a1 == A) & (a2 == B)
    ca, _ = cfit(z[s & half], [G[s & half], S[s & half], Tq[s & half]])
    cb, _ = cfit(z[s & ~half], [G[s & ~half], S[s & ~half], Tq[s & ~half]])
    dmax = max(dmax, abs(ca - cb).max())
print(f"per-cell 3-term fits (a1>=1): largest |coefficient difference| between the two halves of the data: {dmax:.4f}")

# out-of-sample EV of the per-cell 3-term model: fit on first half, score on second half
res = 0.0; ss = 0.0
for (A, B), c in cells.items():
    if A == 0: continue
    s = ok & (a1 == A) & (a2 == B)
    ca, _ = cfit(z[s & half], [G[s & half], S[s & half], Tq[s & half]])
    te = s & ~half
    res += (abs(z[te] - np.column_stack([G[te], S[te], Tq[te]]) @ ca) ** 2).sum(); ss += (abs(z[te]) ** 2).sum()
print(f"per-cell 3-term model fitted on seeds 1100-1101, scored on seeds 1102-1104 (a1>=1): EV = {1 - res / ss:.4f}")

print("\nphase error around phi3(a1) (phase of the 3-term a1-only alpha): percentiles 5/25/50/75/95, share within 21.4 deg")
phi3_arr = np.array([phi3[A] for A in range(10)])
err3 = wrap(deg(z) - np.degrees(phi3_arr[a1]))
for A in range(1, 10):
    s = ok & (a1 == A)
    p = np.percentile(err3[s], qs)
    print(f"  a1={A}: " + " ".join(f"{v:+7.1f}" for v in p) + f"   within 21.4: {np.mean(abs(err3[s]) < 21.37):.4f}   mean|err| {np.mean(abs(err3[s])):.1f}")
print("by ratio (M - S+)/M, a1 >= 1 pooled:")
for lo, hi in zip(rb[:-1], rb[1:]):
    s = sel & (ratio >= lo) & (ratio < hi)
    p = np.percentile(err3[s], qs)
    print(f"  ratio [{lo:.2f},{min(hi, 1):.2f}): share {s.sum() / sel.sum():.3f}  " + " ".join(f"{v:+7.1f}" for v in p)
          + f"   within 21.4: {np.mean(abs(err3[s]) < 21.37):.4f}   mean|err| {np.mean(abs(err3[s])):.1f}")
print("share within 21.4 deg, a1 x ratio bin;  then median error")
for A in range(1, 10):
    row = []; row2 = []
    for lo, hi in zip(rb[:-1], rb[1:]):
        s = ok & (a1 == A) & (ratio >= lo) & (ratio < hi)
        row.append(f"{np.mean(abs(err3[s]) < 21.37):.3f}"); row2.append(f"{np.median(err3[s]):+6.1f}")
    print(f"  a1={A}: " + "  ".join(row) + "   |   " + "  ".join(row2))
print("states with S <= 0 (ratio = 1 exactly): share of a1>=1 states", np.mean(S[sel] == 0), " within 21.4:", np.mean(abs(err3[sel & (S == 0)]) < 21.37),
      "; ratio in [0.75,1) with S > 0: within 21.4:", np.mean(abs(err3[sel & (S > 0) & (ratio >= 0.75)]) < 21.37))

np.savez(HIDE / "r1_C_clock" / "c1_laws.npz", phi=np.array([phi[A] for A in range(10)]), alpha=np.array([alpha1[A] for A in range(10)]),
         phi3=phi3_arr, alpha3=np.array([alpha3[A] for A in range(10)]),
         cell_alpha=np.array([[cells[(A, B)]["c3"][0] if (A, B) in cells else np.nan for B in range(-1, 10)] for A in range(10)]),
         cell_beta=np.array([[cells[(A, B)]["c3"][1] if (A, B) in cells else np.nan for B in range(-1, 10)] for A in range(10)]),
         cell_gamma=np.array([[cells[(A, B)]["c3"][2] if (A, B) in cells else np.nan for B in range(-1, 10)] for A in range(10)]))
