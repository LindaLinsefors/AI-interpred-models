"""d10: structure of the code found in d8/d9,  c / M = A(a_M, a_S) + B(a_M, a_S) S/M   (model E3; c = tape coordinates).
Quiet states of all steps t = 3..9 pooled; fitted on the first 75% of the sequences, tested on the rest.
A. Simpler forms of E3, each scored by variance explained AND by the functional test of d9 (tape replaced at step t,
   run continued, final answer against the true second argmax):
     E3      one (A, B) per pair (a_M, a_S)                                         56 x 14 numbers
     ADD     A = A1(a_M) + A2(a_S), B = B1(a_M) + B2(a_S)   (two superposed marks)   (8 + 8) x 14
     ADD2    ADD fitted separately for 'S before M' and 'S after M'
     FREE2   each mark is written once and then only multiplied by the eigenvalues:  z_m / M = lam_m^a_S (u_m + u'_m S/M)
             + lam_m^a_M (v_m + v'_m S/M), fitted separately for the two orders      2 x 28 numbers
B. Decoding the VALUES given the ages: within each class (a_M, a_S), least squares of M and of S on the 7 raw tape
   coordinates (no intercept).  R2 on test states, against the same from the rest's 9 coordinates.
C. E3 and E4 of d9 split by what happens after step t.
Output: d10_structure.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
n = 300_000; ntr = 225_000
x = sample(n, 21); H = run(x); truth = second_argmax(x); base = answer(H[:, 9])
P = {t: prefix_stats(x, t) for t in range(3, 10)}
rows = [(np.flatnonzero(P[t]["quiet"]), t) for t in range(3, 10)]
seq = np.concatenate([r for r, _ in rows]); step = np.concatenate([np.full(len(r), t) for r, t in rows])
get = lambda k: np.concatenate([P[t][k][r] for r, t in rows])
M, S, T3, aM, aS, aT = (get(k) for k in ("M", "S", "T3", "aM", "aS", "aT"))
hq = H[seq, step]; craw = tape_coords(hq); c = craw / M[:, None]; tr = seq < ntr; te = ~tr; one = np.ones(len(seq))
g2 = aM * 10 + aS; g3 = g2 * 10 + aT; bef = aS > aM; r = S / M
var = c[tr].var(0)
out(f"{n} sequences, quiet states at t = 3..9: {len(seq)} (train {tr.sum()}, test {te.sum()})")


def fit_classes(group, V, Y=c, intercept=True):
    X = np.column_stack(([one] if intercept else []) + V); pred = np.full(Y.shape, np.nan)
    ug, inv = np.unique(group, return_inverse=True)
    o = np.argsort(inv, kind="stable"); cut = np.searchsorted(inv[o], np.arange(len(ug) + 1))
    for k in range(len(ug)):
        idx = o[cut[k]:cut[k + 1]]; t = idx[tr[idx]]
        if len(t) >= 40:
            pred[idx] = X[idx] @ np.linalg.lstsq(X[t], Y[t], rcond=None)[0]
    return pred


def lsq(X, Y, mask):
    """least squares on the training states of `mask`, prediction for all states of `mask` (zero elsewhere)."""
    p = np.zeros(Y.shape, Y.dtype); k = mask & tr
    p[mask] = X[mask] @ np.linalg.lstsq(X[k], Y[k], rcond=None)[0]
    return p


oh = lambda a: (a[:, None] == np.arange(2, 10)).astype(float)
Xadd = np.hstack([oh(aM), oh(aS), oh(aM) * r[:, None], oh(aS) * r[:, None]])
ADD = lsq(Xadd, c, np.ones(len(seq), bool))
ADD2 = lsq(Xadd, c, bef) + lsq(Xadd, c, ~bef)
FREE2 = np.zeros_like(c)
for j, nm in enumerate(("alt", "short", "h2", "h3")):
    lam = LAM[nm]; Xf = np.stack([lam ** aS, lam ** aM, r * lam ** aS, r * lam ** aM], 1)
    if nm == "alt":
        FREE2[:, 0] = lsq(Xf.real, c[:, 0], bef) + lsq(Xf.real, c[:, 0], ~bef)
    else:
        z = c[:, 2 * j - 1] + 1j * c[:, 2 * j]
        p = lsq(Xf, z, bef) + lsq(Xf, z, ~bef)
        FREE2[:, 2 * j - 1], FREE2[:, 2 * j] = p.real, p.imag
E3 = fit_classes(g2, [r]); E4 = fit_classes(g3, [r, np.maximum(T3, 0) / M]); fb = np.isnan(E4[:, 0]); E4[fb] = E3[fb]
models = {"E3    (a_M,a_S) classes + S/M": E3, "ADD   A1(a_M) + A2(a_S), + S/M": ADD, "ADD2  the same per order": ADD2, "FREE2 two freely turning marks, per order": FREE2}


def final(hn, k, t):
    return answer(hn) if t == 9 else answer(continue_from(hn, x[seq[k], t + 1:])[:, -1])


out("")
out("A. simpler forms of E3: variance of c/M explained (test), and final answer = true second argmax after replacing the tape at step t")
out(f"{'model':44s} {'var.':>6s}" + "".join(f"{'t = ' + str(t):>8s}" for t in range(3, 10)))
out(f"{'untouched':44s} {'':6s}" + "".join(f"{(base == truth)[seq[te & (step == t)]].mean():8.3f}" for t in range(3, 10)))
for name, p in models.items():
    ev = 1 - (((c - p)[te] ** 2).mean(0) / var).mean()
    acc = [(final(set_tape(hq[k], p[k] * M[k, None]), k, t) == truth[seq[k]]).mean() for t in range(3, 10) for k in [te & (step == t)]]
    out(f"{name:44s} {ev:6.3f}" + "".join(f"{v:8.3f}" for v in acc))
out("   per coordinate (alt, Re/Im short, Re/Im h2, Re/Im h3), variance explained:")
for name, p in models.items():
    out(f"   {name:42s}", np.round(1 - ((c - p)[te] ** 2).mean(0) / var, 3))

out("")
out("B. values from the raw coordinates GIVEN the class (a_M, a_S): per-class least squares without intercept, R2 on test states")
rc = rest_coords(hq)
for nm, y in (("M", M), ("S", S), ("M - S", M - S)):
    pt, pr = fit_classes(g2, [craw], y, False), fit_classes(g2, [rc], y, False)
    ok = te & ~np.isnan(pt)
    r2 = lambda p: 1 - ((y - p)[ok] ** 2).mean() / y[ok].var()
    out(f"   {nm:6s} from the tape {r2(pt):.3f}   from the rest {r2(pr):.3f}")
sm_t = fit_classes(g2, [craw], S, False) / fit_classes(g2, [craw], M, False)
sm_c = fit_classes(g2, [c / np.linalg.norm(c, axis=1, keepdims=True)], r)
ok = te & ~np.isnan(sm_c)
out(f"   S/M: ratio of the two tape estimates, median abs. error {np.median(np.abs(r - sm_t)[ok]):.3f} (no R2: the ratio blows up where the estimate of M is near 0);"
    f"  per-class affine fit on the tape direction c/|c|: R2 {1 - ((r - sm_c)[ok] ** 2).mean() / r[ok].var():.3f} (median abs. error {np.median(np.abs(r - sm_c)[ok]):.3f}); sd of S/M {r[ok].std():.3f}")

out("")
out("C. split by what happens AFTER step t (test states, t = 3..8 pooled): final answer = true second argmax")
k = te & (step < 9); sq, stp = seq[k], step[k]
later = np.where(np.arange(10)[None] > stp[:, None], x[sq], -np.inf)
nA = (later > M[k, None]).sum(1); nB = ((later > S[k, None]) & (later <= M[k, None])).sum(1)
E1 = fit_classes(g2, [])
cases = (("no later input above S_t", (nA == 0) & (nB == 0)), ("later input(s) between S_t and M_t only", (nA == 0) & (nB > 0)),
         ("exactly one later input above M_t, none between", (nA == 1) & (nB == 0)), ("other (several changes)", (nA + nB > 1) & (nA > 0)))
out(f"   {'':50s} {'n':>6s} {'untouched':>10s} {'E1':>7s} {'E3':>7s} {'E4':>7s} {'FREE2':>7s}")
fin = {}
for nm, p in (("E1", E1), ("E3", E3), ("E4", E4), ("FREE2", FREE2)):
    fin[nm] = np.concatenate([final(set_tape(hq[kk], p[kk] * M[kk, None]), kk, t) for t in range(3, 9) for kk in [te & (step == t)]])
for nm, sel in cases:
    out(f"   {nm:50s} {sel.sum():6d} {(base == truth)[sq][sel].mean():10.3f}" + "".join(f"{(fin[m] == truth[sq])[sel].mean():7.3f}" for m in fin))
