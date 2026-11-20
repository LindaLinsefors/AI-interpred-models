"""Follow-ups for both claims.  Seeds 5700.. (I2 pool), 5720.. (I1 pool), 5740.. (uniform inputs), 5760.. (t_r = 9).

 A. I2: is the clock COORDINATE the causal variable, or the wave-unit part of the clock pattern?
 B. I1: remove only the comparators' clock component, at every t0 (including the last step); readouts at t0 = 9
 C. I2: why does P(p0) fall for f > 1?  (comparators left on at the record)
 D. I2 by age of the old max and by t_r; records at the last step (t_r = 9, outside the lead's selection)
 E. I1: the on-manifold analogue - replace the INPUT x_t0 by a real near-tie / record
 F. both claims with Unif(-1, 1) inputs
 G. I1: scaling the comparators by a small c > 0
"""
from v3lib import *

out = Tee("a5_followups.txt")
isC = np.isin(ORD_A, COMP)
Lc, Rc = LEFT["clock"], RIGHT["clock"]

# ====================================================================== A
x, tr, p0, q0, M0, S0, nb = gather_I2(250_000, 5700)
n = len(x); r = np.arange(n); tint = tr - 1
out(f"[A] I2 pool: seeds 5700..{5700 + nb - 1}, n = {n}")
H0 = run(x)
hpre = H0[r, tint]
z0 = hpre[:, ORD_A] @ Lc


def line(a, p0=None, tr=None, q0=None):
    p0 = globals()["p0"] if p0 is None else p0
    tr = globals()["tr"] if tr is None else tr
    q0 = globals()["q0"] if q0 is None else q0
    return f"P(p0) = {np.mean(a == p0):.4f}  P(t_r) = {np.mean(a == tr):.4f}  P(q0) = {np.mean(a == q0):.4f}"


def comp_minnorm(target_factor, clip=False):
    """Smallest change of the four comparators alone that makes the clock coordinate z -> target_factor * z."""
    Bm = np.stack([Lc[isC].real, Lc[isC].imag])           # (2, 4)
    P = np.linalg.pinv(Bm)                                # (4, 2)

    def fn(h, rows):
        z = h[:, ORD_A] @ Lc
        dz = z * (target_factor - 1.0)
        d = np.stack([dz.real, dz.imag], 1) @ P.T          # (n, 4)
        h = h.copy()
        h[:, COMP] += d
        return np.maximum(h, 0.0) if clip else h
    return fn


a_cc = Lc[isC] @ Rc[isC]; b_cc = Lc[isC] @ np.conj(Rc[isC])
out(f"    L_comp . R_comp = {a_cc:.3f} (so L_wave . R_wave = {1 - a_cc:.3f});  L_comp . conj(R_comp) = {b_cc:.3f}")
out("    intervention -> median |z'|/|z| of the clock coordinate, median phase shift (deg), and the answer")
tests = [("clock x 0, all 13 units (lead)", mode_fn("clock", f=0)),
         ("clock x 0, wave units only", mode_fn("clock", f=0, units=WAVE)),
         ("clock x 0, comparators only", mode_fn("clock", f=0, units=COMP)),
         ("comparators moved (min norm) so that z' = 0", comp_minnorm(0.0)),
         ("same, clipped at 0", comp_minnorm(0.0, clip=True)),
         ("comparators moved so that z' = 0.5 z", comp_minnorm(0.5)),
         ("comparators moved so that z' = 2 z", comp_minnorm(2.0)),
         ("comparators moved so that z' = z e^{+i w}", comp_minnorm(np.exp(1j * OMEGA))),
         ("comparators moved so that z' = z e^{-i w}", comp_minnorm(np.exp(-1j * OMEGA))),
         ("comparators moved so that z' = -z", comp_minnorm(-1.0)),
         ("wipe the comparators", scale_units(COMP, 0.0)),
         ("clock x 0.5, wave units only", mode_fn("clock", f=0.5, units=WAVE)),
         ("clock k=+1, wave units only", mode_fn("clock", k=1, units=WAVE)),
         ("clock k=-1, wave units only", mode_fn("clock", k=-1, units=WAVE))]
for label, fn in tests:
    h9, hiv = run_iv(x, tint, fn, return_state=True)
    a = answer(h9)
    z1 = hiv[:, ORD_A] @ Lc
    ratio = np.median(np.abs(z1) / np.abs(z0))
    dph = np.degrees(np.angle(np.mean(z1 * np.conj(z0) / np.abs(z0) ** 2)))
    out(f"    {label:46s} |z'|/|z| = {ratio:.3f}  phase {dph:+7.1f}  | {line(a)}  p0-1: {np.mean(a == p0 - 1):.3f}  p0+1: {np.mean(a == p0 + 1):.3f}"
        f"  neg = {np.mean((hiv < -1e-9).any(1)):.2f}")

# ====================================================================== C
out("\n[C] f > 1: comparators still on at the record step t_r, and P(p0) given all four are off there")
for f in (0.5, 1, 1.25, 1.5, 2, 3):
    for clip in (False, True):
        B = n
        h = np.zeros((B, N)); htr = np.zeros((B, N))
        fn = mode_fn("clock", f=f, clip=clip)
        for t in range(T):
            h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
            rows = np.flatnonzero(tint == t)
            if rows.size:
                h[rows] = fn(h[rows], rows)
            rows = np.flatnonzero(tr == t)
            if rows.size:
                htr[rows] = h[rows]
        a = answer(h)
        off = (htr[:, COMP] == 0).all(1)
        out(f"    f = {f:<4} {'clipped' if clip else 'raw    '}: all four comparators off at t_r in {off.mean():.3f};  unit 4 fires at t_r in {np.mean(htr[:, 4] > 0):.3f};"
            f"  P(p0 | all off) = {np.mean(a[off] == p0[off]):.4f}   P(p0 | some on) = {np.mean(a[~off] == p0[~off]) if (~off).any() else float('nan'):.4f}"
            f"   median h4(t_r)/(10.157 (x_tr - M0)) = {np.median(htr[:, 4] / (10.157 * (x[r, tr] - M0))):.3f}")

# ====================================================================== D
out("\n[D] I2 by age of the old max at t_r - 1 (rows) : n, then P(p0), P(t_r) for f = 0 | P(p0) for f = 0.5 | k=+1: P(p0), P(p0-1) | k=-1: P(p0), P(p0+1)")
age = tr - 1 - p0
res = {key: answer(run_iv(x, tint, fn)) for key, fn in (("f0", mode_fn("clock", f=0)), ("f.5", mode_fn("clock", f=0.5)),
                                                         ("k+1", mode_fn("clock", k=1)), ("k-1", mode_fn("clock", k=-1)))}
for g in range(2, 8):
    m = age == g
    out(f"    age {g}: n={m.sum():6d}  f=0: {np.mean(res['f0'][m] == p0[m]):.3f} {np.mean(res['f0'][m] == tr[m]):.3f} | f=0.5: {np.mean(res['f.5'][m] == p0[m]):.3f}"
        f" | k=+1: {np.mean(res['k+1'][m] == p0[m]):.3f} {np.mean(res['k+1'][m] == p0[m] - 1):.3f} | k=-1: {np.mean(res['k-1'][m] == p0[m]):.3f} {np.mean(res['k-1'][m] == p0[m] + 1):.3f}")
out("    by t_r:")
for g in range(3, 9):
    m = tr == g
    out(f"    t_r {g}: n={m.sum():6d}  f=0: {np.mean(res['f0'][m] == p0[m]):.3f} {np.mean(res['f0'][m] == tr[m]):.3f} | f=0.5: {np.mean(res['f.5'][m] == p0[m]):.3f}"
        f" | k=+1: {np.mean(res['k+1'][m] == p0[m]):.3f} {np.mean(res['k+1'][m] == p0[m] - 1):.3f} | k=-1: {np.mean(res['k-1'][m] == p0[m]):.3f} {np.mean(res['k-1'][m] == p0[m] + 1):.3f}")
out("    by relative gap of the old top two, (M0 - S0)/M0, f = 0.5: P(p0) in bins (0,.1], (.1,.2], (.2,.4], (.4,.7], (.7,1): "
    + " ".join(f"{np.mean(res['f.5'][(g_ > a_) & (g_ <= b_)] == p0[(g_ > a_) & (g_ <= b_)]):.3f}" for g_ in [(M0 - S0) / M0] for a_, b_ in ((0, .1), (.1, .2), (.2, .4), (.4, .7), (.7, 1))))
out("    by size of the record, (x_tr - M0)/M0, f = 0.5: P(p0) in bins (0,.1], (.1,.3], (.3,.6], (.6,1], >1: "
    + " ".join(f"{np.mean(res['f.5'][(g_ > a_) & (g_ <= b_)] == p0[(g_ > a_) & (g_ <= b_)]):.3f}" for g_ in [(x[r, tr] - M0) / M0] for a_, b_ in ((0, .1), (.1, .3), (.3, .6), (.6, 1), (1, np.inf))))

out("\n    records at the LAST step (t_r = 9; not in the lead's selection; same other conditions), seeds 5760..5763")
xs = []
for s in range(5760, 5764):
    xx = sample(400_000, s)
    o = np.argsort(xx, axis=1)
    pre = np.argsort(xx[:, :9], axis=1)
    rr = np.arange(len(xx))
    pp, qq = pre[:, -1], pre[:, -2]
    ok = (o[:, -1] == 9) & (8 - pp >= 2) & (xx[rr, qq] > 0.2) & (xx[rr, pp] - xx[rr, qq] > 0.1)
    xs.append((xx[ok], pp[ok], qq[ok]))
x9 = np.concatenate([v[0] for v in xs]); p9 = np.concatenate([v[1] for v in xs]); q9 = np.concatenate([v[2] for v in xs])
t9 = np.full(len(x9), 9)
out(f"    n = {len(x9)};  unperturbed: {line(answer(run(x9)[:, -1]), p9, t9, q9)}")
for label, kw in (("f = 0", dict(f=0)), ("f = 0.5", dict(f=0.5)), ("f = 2", dict(f=2)), ("k = +1", dict(k=1)), ("k = -1", dict(k=-1))):
    a = answer(run_iv(x9, np.full(len(x9), 8), mode_fn("clock", **kw)))
    out(f"    {label:8s}: {line(a, p9, t9, q9)}  p0-1: {np.mean(a == p9 - 1):.3f}  p0+1: {np.mean(a == p9 + 1):.3f}")

# ====================================================================== B
x1, pM, pS, M, S, nb1 = gather_I1(500_000, 5720)
out(f"\n[B] I1 pool: seeds 5720..{5720 + nb1 - 1}, n = {len(x1)}")
out("    P(ans = pM) by k (rows) and t0 (columns 1..9) when only the comparators' CLOCK component is removed (clipped at 0)")
E2 = mode_fn("clock", f=0, units=COMP, clip=True)
for k in range(1, 10):
    v = pM + k <= 9
    t0 = (pM + k)[v]
    a = answer(run_iv(x1[v], t0, E2))
    out(f"    k={k}: " + " ".join(f"{np.mean(a[t0 == t] == pM[v][t0 == t]):.4f}" if (t0 == t).any() else "  -   " for t in range(1, 10))
        + f"   | pooled {np.mean(a == pM[v]):.4f}  P(runner-up) {np.mean(a == pS[v]):.4f}  P(ans = t0) {np.mean(a == t0):.4f}")
out("    readouts at the last step (t0 = 9), by age k of the max: P(argmax = pM) for")
out("    (i) logits without the comparator columns (= wipe at t0 = 9); (ii) the same, position 9 excluded; (iii) full logits minus the")
out("    comparators' clock component; (iv) full logits, position of the true runner-up excluded")
H1 = run(x1)[:, -1]
full = H1 @ U.T
mm = full - H1[:, COMP] @ U[:, COMP].T
zc = H1[:, ORD_A] @ Lc
dclk = 2 * np.real(np.multiply.outer(zc, Rc))[:, isC]            # clock component on the comparators
s3 = full - dclk @ U[:, COMP].T
for k in range(1, 10):
    v = pM == 9 - k
    mm2 = mm[v].copy(); mm2[:, 9] = -np.inf
    f4 = full[v].copy(); f4[np.arange(v.sum()), pS[v]] = -np.inf
    out(f"    k={k}: (i) {np.mean(mm[v].argmax(1) == pM[v]):.4f}  (ii) {np.mean(mm2.argmax(1) == pM[v]):.4f}  (iii) {np.mean(s3[v].argmax(1) == pM[v]):.4f}"
        f"  (iv) {np.mean(f4.argmax(1) == pM[v]):.4f}")

# ====================================================================== E
out("\n[E] on-manifold analogue of the wipe: replace the INPUT x_t0 (t0 = pM + k <= 7) by c * M and run the unmodified network.")
out("    P(ans = pM) | P(ans = t0);  k = 1..6")
for c in (0.9, 0.97, 0.99, 1.0, 1.01, 1.03, 1.1, 1.3):
    row = []
    for k in range(1, 7):
        v = pM + k <= 7
        xv = x1[v].copy(); t0 = (pM + k)[v]
        xv[np.arange(v.sum()), t0] = c * M[v]
        a = answer(run(xv)[:, -1])
        row.append(f"{np.mean(a == pM[v]):.3f}|{np.mean(a == t0):.3f}")
    out(f"    x_t0 := {c:<4} M : " + "  ".join(row))
row = []
for k in range(1, 7):
    v = pM + k <= 7
    a = answer(run_iv(x1[v], (pM + k)[v], scale_units(COMP, 0.0)))
    row.append(f"{np.mean(a == pM[v]):.3f}|{np.mean(a == (pM + k)[v]):.3f}")
out("    wipe (for comparison): " + "  ".join(row))

# ====================================================================== G
out("\n[G] scale the comparators by a small c (t0 <= 7): P(ans = pM) | P(ans = t0), k = 1..6")
for c in (0, 0.01, 0.02, 0.05, 0.1, 0.2):
    row = []
    for k in range(1, 7):
        v = pM + k <= 7
        a = answer(run_iv(x1[v], (pM + k)[v], scale_units(COMP, c)))
        row.append(f"{np.mean(a == pM[v]):.3f}|{np.mean(a == (pM + k)[v]):.3f}")
    out(f"    c = {c:<5}: " + "  ".join(row))

# ====================================================================== F
out("\n[F] the two claims with Unif(-1, 1) inputs (same thresholds), seeds 5740..")
xs, pMs, pSs = [], [], []
for s in range(5740, 5744):
    xx = sample(400_000, s, dist="uniform")
    ok, a_, b_, _, _ = select_I1(xx)
    xs.append(xx[ok]); pMs.append(a_[ok]); pSs.append(b_[ok])
xu, pMu, pSu = np.concatenate(xs), np.concatenate(pMs), np.concatenate(pSs)
out(f"    I1 selection: n = {len(xu)} of 1600000;  unperturbed P(ans = runner-up) = {np.mean(answer(run(xu)[:, -1]) == pSu):.4f}")
for name, units in (("comparators", COMP), ("control 10,11,12,15", [10, 11, 12, 15])):
    row, row2 = [], []
    for k in range(1, 7):
        v = pMu + k <= 9
        a = answer(run_iv(xu[v], (pMu + k)[v], scale_units(units, 0.0)))
        row.append(f"{np.mean(a == pMu[v]):.3f}"); row2.append(f"{np.mean(a == pSu[v]):.3f}")
    out(f"    wipe {name:20s}: pooled over valid t0, k = 1..6: P(pM) = " + " ".join(row) + "   P(runner-up) = " + " ".join(row2))
v = pMu + 2 <= 9
a = answer(run_iv(xu[v], (pMu + 2)[v], scale_units(COMP, 0.0)))
t0 = (pMu + 2)[v]
out("    k = 2 by t0 = 2..9: " + " ".join(f"{np.mean(a[t0 == t] == pMu[v][t0 == t]):.4f}" for t in range(2, 10)))
xs = []
for s in range(5750, 5760):
    xx = sample(400_000, s, dist="uniform")
    ok, a_, b_, c_, _, _ = select_I2(xx)
    xs.append((xx[ok], a_[ok], b_[ok], c_[ok]))
xu = np.concatenate([v[0] for v in xs]); tru = np.concatenate([v[1] for v in xs]); p0u = np.concatenate([v[2] for v in xs]); q0u = np.concatenate([v[3] for v in xs])
out(f"    I2 selection: n = {len(xu)} of 4000000; unperturbed {line(answer(run(xu)[:, -1]), p0u, tru, q0u)}")
for label, kw in (("f = 0", dict(f=0)), ("f = 0.25", dict(f=0.25)), ("f = 0.5", dict(f=0.5)), ("f = 0.75", dict(f=0.75)), ("f = 1.5", dict(f=1.5)), ("f = 2", dict(f=2)),
                  ("k = -2", dict(k=-2)), ("k = -1", dict(k=-1)), ("k = +1", dict(k=1)), ("k = +2", dict(k=2))):
    a = answer(run_iv(xu, tru - 1, mode_fn("clock", **kw)))
    out(f"    {label:8s}: {line(a, p0u, tru, q0u)}  " + " ".join(f"p0{j:+d}: {np.mean(a - p0u == j):.3f}" for j in (-2, -1, 1, 2)))
