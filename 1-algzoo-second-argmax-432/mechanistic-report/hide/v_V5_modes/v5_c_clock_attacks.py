"""Verifier V5, part C: attacks on the clock interpretation (M5).

 C1  resting pattern: how often, and the clock law on states that are / are not in it
 C2  per-step rotation z_{t+1}/z_t on exact-linear (resting) steps vs clipped steps; runner-up events
 C3  phase law phi(a1) and its spread
 C4  radius law: M - S  vs  M - c S  vs  M alone; S <= 0; gap at the time of the record
Seeds 5010-5013 (4 x 200k sequences).   Run:  python -B v5_c_clock_attacks.py > v5_c_clock_attacks.txt
"""
from v5_lib import *

lam = LAM["clock"]
Lw = LEFT["clock"] @ w[ORD]                 # what input x adds to z on an exact linear step
acc = {}
def add(key, arr):
    arr = np.asarray(arr)
    if arr.dtype.kind == "i":
        arr = arr.astype(np.int8)                           # ages, counts, t: all < 128
    acc.setdefault(key, []).append(arr)

lin_err = 0.0
for seed in SEEDS:
    x, H, Z, st, m = batch(seed)
    B = x.shape[0]
    z = m["clock"] / W4
    on13 = (Z[:, :, ORD] > 0).all(-1)
    noff = 13 - (Z[:, :, ORD] > 0).sum(-1)
    poff = (Z[:, :, PULSE] <= 0).all(-1)
    rest = on13 & poff                                       # resting pattern at step t
    lin = np.zeros_like(rest)                                # exact linear step (t-1) -> t for the ordinary units
    lin[:, 1:] = on13[:, 1:] & (H[:, :-1][:, :, PULSE] == 0).all(-1)
    run_lin = np.zeros(rest.shape, dtype=int)                # number of consecutive exact-linear steps ending at t
    for t in range(1, T):
        run_lin[:, t] = np.where(lin[:, t], run_lin[:, t - 1] + 1, 0)
    # sanity: on exact linear steps z_t = lam z_{t-1} + (L.w) x_t
    pred = lam * z[:, :-1] + Lw * x[:, 1:] / W4
    lin_err = max(lin_err, np.abs(pred - z[:, 1:])[lin[:, 1:]].max())
    # gap at the time of the record: runner-up of the prefix at step p1 (= max of x_0..x_{p1-1})
    S_at_rec = np.take_along_axis(st["S"], np.maximum(st["pM"], 0), 1)       # S at step p1 (window incl. the record)
    tt = np.broadcast_to(np.arange(T)[None, :], rest.shape)
    for k, a in (("M", st["M"]), ("S", st["S"]), ("a1", st["a1"]), ("a2", st["a2"]), ("t", tt), ("z", z),
                 ("rest", rest), ("on13", on13), ("poff", poff), ("noff", noff), ("lin", lin), ("run_lin", run_lin),
                 ("Srec", S_at_rec), ("T3", st["T3"])):
        add(k, a.reshape(-1))
    # per-step pairs t -> t+1
    for k, a in (("zn", z[:, 1:]), ("z0", z[:, :-1]), ("xn", x[:, 1:]), ("M0", st["M"][:, :-1]), ("S0", st["S"][:, :-1]),
                 ("a10", st["a1"][:, :-1]), ("a20", st["a2"][:, :-1]), ("linn", lin[:, 1:]), ("noffn", noff[:, 1:]),
                 ("h4", H[:, :-1, 4]), ("h5", H[:, :-1, 5]), ("h9", H[:, :-1, 9])):
        add("p_" + k, a.reshape(-1))
D = {k: np.concatenate(v) for k, v in acc.items()}
M, S, a1, a2, tt, z, rest, on13, poff, noff, lin, run_lin, Srec = (D[k] for k in
    ("M", "S", "a1", "a2", "t", "z", "rest", "on13", "poff", "noff", "lin", "run_lin", "Srec"))
Sp = np.maximum(S, 0)
print(f"seeds {SEEDS}, {NB} sequences each; {len(M)} states.  max |z_t - (lam z_(t-1) + L.w x_t)| on exact-linear steps = {lin_err:.2e}")

# ------------------------------------------------------------------------------------------------------------
print("=" * 120)
print("C1  RESTING PATTERN (all 13 ordinary units have pre-activation > 0 and pulses 4, 5, 9 have pre-activation <= 0 at step t)")
print(f"   all states: P(resting) = {rest.mean():.4f}   P(all 13 on) = {on13.mean():.4f}   P(pulses off) = {poff.mean():.4f}")
print(f"   states t >= 1 with S > 0.05: P(resting) = {rest[S > 0.05].mean():.4f}")
print("   by a1 (states with S > 0.05):  P(resting)  P(all13 on)  P(pulses off)  mean #ordinary units off  P(exact-linear step into t)  P(>=2 consecutive exact-linear steps ending at t)  P(>=3)  P(>=4)")
ok = S > 0.05
for A1 in range(0, 10):
    s = ok & (a1 == A1)
    print(f"   a1={A1}: {rest[s].mean():.4f}   {on13[s].mean():.4f}   {poff[s].mean():.4f}   {noff[s].mean():.3f}   {lin[s].mean():.4f}   {(run_lin[s] >= 2).mean():.4f}   {(run_lin[s] >= 3).mean():.4f}   {(run_lin[s] >= 4).mean():.4f}   n={int(s.sum())}")
print("   longest run of consecutive exact-linear steps in a sequence: distribution over all sequences (max over t of run length):",
      np.round(np.bincount(run_lin.reshape(-1, T).max(1).astype(int), minlength=10) / (len(run_lin) // T), 4))
print("   by t (all states): P(resting):", np.round([rest[tt == t0].mean() for t0 in range(T)], 3))
print("   at t = 9: P(resting) =", round(rest[tt == 9].mean(), 4), " P(all 13 on) =", round(on13[tt == 9].mean(), 4))

print("\n   clock law z/w4 ~ alpha (M-S) + beta S per a1 (pooled over a2, S > 0.05), fitted on ALL states of the a1 bin, then scored separately:")
print("   a1 | frac resting | R2u all | R2u resting | R2u non-resting | R2c resting | R2c non-resting || own fit resting: alpha, beta, R2u | own fit non-resting: alpha, beta, R2u")
X = np.stack([M - S, S], 1)
def r2(y, yhat):
    ss = (np.abs(y - yhat) ** 2).sum()
    return 1 - ss / (np.abs(y) ** 2).sum(), 1 - ss / (np.abs(y - y.mean()) ** 2).sum()
for A1 in range(1, 10):
    s = ok & (a1 == A1)
    c, r2u, r2c = lsq(X[s], z[s])
    sr, sn = s & rest, s & ~rest
    out = f"   a1={A1} | {rest[s].mean():.3f} | {r2u:.4f} |"
    for sub in (sr, sn):
        out += f" {r2(z[sub], X[sub] @ c)[0]:.4f} |" if sub.sum() > 500 else "   n/a  |"
    for sub in (sr, sn):
        out += f" {r2(z[sub], X[sub] @ c)[1]:.4f} |" if sub.sum() > 500 else "   n/a  |"
    for sub in (sr, sn):
        if sub.sum() > 500:
            cc, ru, rc = lsq(X[sub], z[sub])
            out += f"| {pol(cc[0])}, {pol(cc[1])}, {ru:.4f} "
        else:
            out += "| n/a "
    print(out)
print("\n   same split, but only cells with a2 > a1 (no runner-up event since the record), gap-only model z/w4 ~ alpha (M - S):")
for A1 in range(1, 9):
    s = ok & (a1 == A1) & (a2 > A1)
    out = f"   a1={A1}: frac resting {rest[s].mean():.3f}"
    for nm, sub in (("all", s), ("resting", s & rest), ("non-resting", s & ~rest), ("#off>=2", s & (noff >= 2))):
        if sub.sum() > 500:
            cc, ru, rc = lsq((M - S)[sub, None], z[sub])
            out += f" | {nm}: alpha={pol(cc[0])} R2u={ru:.4f} R2c={rc:.4f} n={int(sub.sum())}"
    print(out)

# which ordinary units are off, by a1 (states with S > 0.05, gap/M > 0.3)
print("\n   P(unit pre-activation <= 0) by a1, states with S > 0.05 and (M-S)/M > 0.3: (computed on seed 5010 only)")
x, H, Z, st, m = batch(SEEDS[0])
sel0 = (st["S"] > 0.05) & ((st["M"] - st["S"]) / st["M"] > 0.3)
print("   a1   " + " ".join(f"u{u:<4d}" for u in ORD) + "  | pulses 4, 5, 9 ON")
for A1 in range(0, 10):
    s = sel0 & (st["a1"] == A1)
    off = (Z[s][:, ORD] <= 0).mean(0)
    pon = (Z[s][:, PULSE] > 0).mean(0)
    print(f"   {A1}    " + " ".join(f"{v:5.2f}" for v in off) + "  | " + " ".join(f"{v:5.2f}" for v in pon))
# eigenvalue nearest to the clock for the linear maps of the most common gate patterns (pulses off)
print("\n   most common ordinary-unit gate patterns (states with S > 0.05, a1 >= 2; seed 5010) and the eigenvalue of D A closest to the clock's:")
s = (st["S"] > 0.05) & (st["a1"] >= 2)
G = (Z[s][:, ORD] > 0)
pid = (G * (1 << np.arange(13))).sum(1)
vals, cnt = np.unique(pid, return_counts=True)
top = np.argsort(-cnt)[:12]
A = W[np.ix_(ORD, ORD)]
for k in top:
    gate = ((vals[k] >> np.arange(13)) & 1).astype(float)
    ev = np.linalg.eigvals(gate[:, None] * A)
    e = ev[np.argmin(np.abs(ev - lam))]
    offu = [ORD[i] for i in range(13) if gate[i] == 0]
    print(f"   freq {cnt[k] / s.sum():.3f}  units off {str(offu):22s}  nearest eigenvalue: |lam| = {abs(e):.4f}, angle = {np.degrees(np.angle(e)):6.2f} deg")

# ------------------------------------------------------------------------------------------------------------
print("=" * 120)
print("C2  PER-STEP ROTATION  rho = z_(t+1) / z_t   (steps with M_t > 0.3 and M_t - max(S_t, 0) > 0.2)")
P = {k[2:]: v for k, v in D.items() if k.startswith("p_")}
zn, z0, xn, M0, S0, a10, a20, linn, noffn = (P[k] for k in ("zn", "z0", "xn", "M0", "S0", "a10", "a20", "linn", "noffn"))
base = (M0 > 0.3) & (M0 - np.maximum(S0, 0) > 0.2) & (a10 >= 1)
noev = base & (xn < S0)
print(f"   eigenvalue: |lam| = {abs(lam):.4f}, angle = {np.degrees(np.angle(lam)):.2f} deg;  input leak on a linear step: (L.w)/w4 = {pol(Lw / W4)} per unit x")
print("   no-event steps (x_(t+1) < S_t): by age a1 at t (step a1 -> a1+1). 'slope' = sum(z1 conj z0)/sum|z0|^2 (complex LS through origin)")
print("   a1->  | frac exact-linear | exact-linear: slope |.|@deg, median angle [q25,q75], median |rho| | clipped (not exact-linear): same | all: same")
def stepstats(s):
    if s.sum() < 500:
        return "n/a"
    rho = zn[s] / z0[s]
    sl = (zn[s] * np.conj(z0[s])).sum() / (np.abs(z0[s]) ** 2).sum()
    ang = np.degrees(np.angle(rho))
    q = np.percentile(ang, [25, 50, 75])
    return f"slope {pol(sl)}  ang med {q[1]:5.1f} [{q[0]:5.1f},{q[2]:5.1f}]  |rho| med {np.median(np.abs(rho)):.3f}  n={int(s.sum())}"
for A1 in range(1, 9):
    s = noev & (a10 == A1)
    print(f"   {A1}->{A1 + 1} | {linn[s].mean():.3f} | {stepstats(s & linn)} | {stepstats(s & ~linn)} | {stepstats(s)}")
s = noev & (a10 >= 2)
print(f"   pooled a1 >= 2: exact-linear: {stepstats(s & linn)}")
print(f"                   clipped     : {stepstats(s & ~linn)}")
print(f"                   all         : {stepstats(s)}")
print("   clipped steps by number of ordinary units off at t+1 (a1 >= 2 at t):")
for k in (1, 2, 3, 4):
    print(f"      #off = {k}: {stepstats(noev & (a10 >= 2) & ~linn & (noffn == k))}")

print("\n   runner-up event at t+1 (S_t < x_(t+1) < M_t, S_t > 0.05, a1 >= 1): q = (M - x_(t+1)) / (M - S_t) = new gap / old gap")
print("   if the radius tracks the CURRENT gap, |rho| ~ q * (no-event |rho|); if it keeps the old gap, |rho| does not depend on q")
ev = base & (xn > S0) & (xn < M0) & (S0 > 0.05)
q = (M0 - xn) / (M0 - S0)
for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0)):
    s = ev & (q > lo) & (q <= hi) & (a10 >= 2)
    rho = zn[s] / z0[s]
    ang = np.degrees(np.angle(rho))
    print(f"   q in ({lo:.2f},{hi:.2f}]: median q = {np.median(q[s]):.3f}  median |rho| = {np.median(np.abs(rho)):.3f}  median angle = {np.median(ang):6.1f} [{np.percentile(ang, 25):6.1f},{np.percentile(ang, 75):6.1f}]  n={int(s.sum())}")
s = ev & (a10 >= 2)
Xq = np.stack([q[s], np.ones(s.sum())], 1)
cq, *_ = np.linalg.lstsq(Xq, np.abs(zn[s] / z0[s]), rcond=None)
print(f"   LS |rho| ~ {cq[0]:.3f} q + {cq[1]:.3f}   (tracks current gap: slope ~0.95-1, intercept ~0;  keeps old gap: slope 0, intercept ~0.95-1)")

# ------------------------------------------------------------------------------------------------------------
print("=" * 120)
print("C3  PHASE LAW  phi(a1) = phase of pooled alpha (z/w4 ~ alpha (M-S) + beta S, S > 0.05)")
phi = {}
for A1 in range(1, 10):
    s = ok & (a1 == A1)
    c, _, _ = lsq(X[s], z[s])
    phi[A1] = np.angle(c[0])
ph = np.unwrap([phi[a] for a in range(1, 10)])
aa = np.arange(1, 10)
for lo in (1, 2):
    sl, ic = np.polyfit(aa[lo - 1:], np.degrees(ph[lo - 1:]), 1)
    res = np.degrees(ph[lo - 1:]) - (sl * aa[lo - 1:] + ic)
    print(f"   linear fit of phi over a1 = {lo}..9: slope = {sl:.2f} deg/step, phi(a1=1) intercept = {sl + ic:.2f} deg, max |residual| = {np.abs(res).max():.2f} deg")
print("   phi(a1) deg:", np.round(np.degrees(ph), 1), "  steps:", np.round(np.diff(np.degrees(ph)), 1))
print("   phase of the raw state: delta = arg z - phi(a1). States with M > 0.3, by relative gap r = (M - max(S,0))/M")
print("   a1 | r in (0,0.1]: median|delta|, P(|delta|<21.4) | (0.1,0.3] | (0.3,0.6] | (0.6,1.0] || all r: circ-mean delta, median |delta|, 90th pct, P(|delta|<21.4)")
r = (M - Sp) / M
for A1 in range(1, 10):
    s0 = (a1 == A1) & (M > 0.3)
    out = f"   a1={A1} |"
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)):
        s = s0 & (r > lo) & (r <= hi)
        d = np.degrees(wrap(np.angle(z[s]) - phi[A1]))
        out += f" {np.median(np.abs(d)):5.1f}, {np.mean(np.abs(d) < 21.37):.3f} |"
    d = np.degrees(wrap(np.angle(z[s0]) - phi[A1]))
    cm = np.degrees(np.angle(np.exp(1j * np.radians(d)).mean()))
    out += f"| {cm:5.1f}, {np.median(np.abs(d)):5.1f}, {np.percentile(np.abs(d), 90):5.1f}, {np.mean(np.abs(d) < 21.37):.3f}"
    print(out)
print("   decoding the age of the max from the clock phase alone: a_hat = argmin_a |arg z - phi(a)| over a = 1..t   (states with a1 >= 1, M > 0.3)")
phis = np.array([phi[a] for a in range(1, 10)])
s0 = (a1 >= 1) & (M > 0.3)
dist = np.abs(wrap(np.angle(z[s0])[:, None] - phis[None, :]))
dist[np.arange(1, 10)[None, :] > tt[s0][:, None]] = np.inf
ahat = dist.argmin(1) + 1
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (0, 1.0)):
    sb = (r[s0] > lo) & (r[s0] <= hi)
    print(f"      r in ({lo},{hi}]: P(a_hat = a1) = {np.mean(ahat[sb] == a1[s0][sb]):.3f}   by a1 = 1..9: "
          + " ".join(f"{np.mean(ahat[sb & (a1[s0] == A1)] == A1):.2f}" for A1 in range(1, 10)))

# ------------------------------------------------------------------------------------------------------------
print("=" * 120)
print("C4  RADIUS LAW")
print("   (a) binned median of |z| / (w4 M) against S/M (states with M > 0.3, a1 fixed).  M - S predicts 1 - S/M;  M alone predicts a constant")
bins = [(-np.inf, -1), (-1, -0.3), (-0.3, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
print("   S/M bin:        " + " ".join(f"{('(%.2g,%.2g]' % b):>12s}" for b in bins))
rs = S / M
for A1 in (1, 2, 3, 4, 6, 8):
    s0 = (a1 == A1) & (M > 0.3)
    out = f"   a1={A1} med |z|/(w4 M)"
    for lo, hi in bins:
        s = s0 & (rs > lo) & (rs <= hi)
        out += f" {np.median(np.abs(z[s]) / M[s]):12.3f}" if s.sum() > 300 else f" {'n/a':>12s}"
    print(out)
    out = f"   a1={A1} med Re(z e^-iphi)/(w4 M)"[:26]
    for lo, hi in bins:
        s = s0 & (rs > lo) & (rs <= hi)
        out += f" {np.median((z[s] * np.exp(-1j * phi[A1])).real / M[s]):12.3f}" if s.sum() > 300 else f" {'n/a':>12s}"
    print(out + "   <- in-phase part")
print("\n   (b) real LS on the in-phase part  Re(z e^{-i phi(a1)}) / w4 ~ bM * M - bS * S   (no intercept), states with M > 0.3:")
print("       a1 |  S > 0.05: bM, bS, c = bS/bM, R2u, R2c  |  S < -0.05: bM, bS, R2u  (M - S predicts bS = bM also for S < 0; M - max(S,0) predicts bS = 0)  | S<-0.05: M alone R2u")
for A1 in range(1, 10):
    out = f"       a1={A1} |"
    y = (z * np.exp(-1j * phi[A1])).real
    s = (a1 == A1) & (M > 0.3) & (S > 0.05)
    c, ru, rc = lsq(np.stack([M[s], -S[s]], 1), y[s])
    out += f" {c[0]:.3f}, {c[1]:.3f}, c = {c[1] / c[0]:.3f}, {ru:.4f}, {rc:.4f} |"
    s = (a1 == A1) & (M > 0.3) & (S < -0.05) & np.isfinite(S)
    if s.sum() > 500:
        c, ru, rc = lsq(np.stack([M[s], -S[s]], 1), y[s])
        c1, ru1, _ = lsq(M[s, None], y[s])
        out += f" {c[0]:.3f}, {c[1]:.3f}, {ru:.4f} | {c1[0]:.3f}, {ru1:.4f}  n={int(s.sum())}"
    print(out)
print("\n   (c) one-parameter-family comparison, complex fit z/w4 ~ alpha (M - c S) per a1 (S > 0.05, pooled over a2): uncentred R2 for c = 0, 0.5, 0.8, 0.87, 1.0 and the best real c")
for A1 in range(1, 10):
    s = ok & (a1 == A1)
    out = f"       a1={A1}:"
    best = (None, -1)
    for c in np.round(np.arange(0, 1.201, 0.01), 2):
        _, ru, rc = lsq((M[s] - c * S[s])[:, None], z[s])
        if ru > best[1]:
            best = (c, ru, rc)
        if c in (0.0, 0.5, 0.8, 0.87, 1.0):
            out += f"  c={c:.2f}: {ru:.4f} (centred {rc:.3f})"
    print(out + f"   best c = {best[0]:.2f}: {best[1]:.4f} (centred {best[2]:.3f})")

print("\n   (d) current gap or gap at the time of the record?  Cells with a2 < a1 (runner-up arrived after the max), S > 0.05, S_rec+ = max(runner-up at record time, 0):")
print("       complex LS  z/w4 ~ alpha (M - S) + gamma (S - S_rec+).   current gap: gamma = 0;   frozen gap at record: gamma = alpha")
Srp = np.maximum(Srec, 0)
for A1 in range(2, 10):
    out = f"       a1={A1}:"
    for nm, s in (("a2=0", ok & (a1 == A1) & (a2 == 0)), ("a2=1", ok & (a1 == A1) & (a2 == 1)), ("1<a2<a1", ok & (a1 == A1) & (a2 > 1) & (a2 < A1)), ("all a2<a1", ok & (a1 == A1) & (a2 < A1))):
        if s.sum() < 2000:
            continue
        Xd = np.stack([M[s] - S[s], S[s] - Srp[s]], 1)
        c, ru, rc = lsq(Xd, z[s])
        ratio = c[1] / c[0]
        out += f"  [{nm}] alpha={pol(c[0])} gamma/alpha={ratio.real:+.3f}{ratio.imag:+.3f}i R2u={ru:.4f}"
    print(out)
print("       three-regressor complex LS z/w4 ~ alpha (M - S) + beta S + gamma (S - S_rec+), all cells a2 < a1 pooled per a1:")
for A1 in range(2, 9):                                      # a1 = 9: the max is x_0, S_rec+ = 0, regressors collinear
    s = ok & (a1 == A1) & (a2 < A1)
    Xd = np.stack([M[s] - S[s], S[s], S[s] - Srp[s]], 1)
    c, ru, rc = lsq(Xd, z[s])
    c2, ru2, rc2 = lsq(Xd[:, :2], z[s])
    print(f"       a1={A1}: alpha={pol(c[0])} beta={pol(c[1])} gamma={pol(c[2])}  R2u={ru:.4f} (without gamma: {ru2:.4f});  centred {rc:.4f} (without gamma {rc2:.4f})")
