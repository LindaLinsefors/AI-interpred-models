"""I4 (what decides near ties).  Input sweeps: move ONE input x_l (the later of two candidates) through the value
of an earlier one x_e and find where the net's preference between the two positions flips.

  runner-up vs third ('RT'):  e and l are the positions of S and T (either order of size); M elsewhere, M - S >= 0.2 M.
        delta(eps) = logit[l] - logit[e]  at  x_l = x_e + eps*M.  Ideal: delta > 0  iff  eps > 0  (the larger is the runner-up).
        eps* = takeover offset: delta(eps*) = 0.   eps* > 0: a later input must beat the earlier one by eps* M to win
        (errors when the later one is the true runner-up with gap < eps*);  eps* < 0: a later input already wins when
        it is still smaller (errors when the later one is the third with gap < |eps*|).
  max vs runner-up ('MR'):   e = position of the max, l = a later position, all others < 0.7 M.
        delta(eps) = logit[e] - logit[l]  at  x_l = M (1 + eps).   Ideal: delta > 0 iff eps > 0 (l is the new max, so e is
        the runner-up).  eps* = record offset: the later input is treated as the new max iff eps > eps*.
Grid: eps in [-0.15, 0.15], step 0.0025; eps* = first sign change (linear interpolation), nan if none.
"""
from icommon import *

EPS = np.round(np.arange(-0.15, 0.15001, 0.0025), 5)


def sweep(x, pe, pl, kind, Mv):
    """delta (B, len(EPS)) and answers (B, len(EPS))"""
    B = len(x); r = np.arange(B)
    D = np.empty((B, len(EPS))); A = np.empty((B, len(EPS)), dtype=np.int8)
    for i, e in enumerate(EPS):
        xx = x.copy()
        xx[r, pl] = x[r, pe] + e * Mv
        lg = logits(run(xx))
        D[:, i] = (lg[r, pl] - lg[r, pe]) if kind == "RT" else (lg[r, pe] - lg[r, pl])
        A[:, i] = lg.argmax(1)
    return D, A


def crossing(D):
    """first sign change from <= 0 to > 0 along the grid; returns eps* (nan if no change), and number of sign changes"""
    s = D > 0
    ch = s[:, 1:] != s[:, :-1]
    nch = ch.sum(1)
    first = ch.argmax(1)
    r = np.arange(len(D))
    d0, d1 = D[r, first], D[r, first + 1]
    eps = EPS[first] + (EPS[first + 1] - EPS[first]) * (-d0) / (d1 - d0)
    eps = np.where(nch > 0, eps, np.where(s[:, 0], -np.inf, np.inf))     # always positive -> below the grid; never -> above
    return eps, nch


def summarize(eps, nch, label):
    fin = np.isfinite(eps)
    qs = np.quantile(eps[fin], [.05, .1, .25, .5, .75, .9, .95])
    print(f"{label}: n = {len(eps)}; crossing inside the grid {fin.mean():.4f} (below grid {np.mean(eps == -np.inf):.4f}, above {np.mean(eps == np.inf):.4f}); "
          f"more than one sign change {np.mean(nch > 1):.4f}")
    print("   eps* quantiles 5/10/25/50/75/90/95%: " + " / ".join(f"{v:+.4f}" for v in qs) +
          f";  mean |eps*| = {np.abs(eps[fin]).mean():.4f};  P(eps* > 0) = {np.mean(eps[fin] > 0):.3f}")


def age_table(val, ae, al, fmt="{:+.3f}", agg=np.median, minn=150):
    rows = []
    for i in range(1, 10):
        row = [i]
        for j in range(0, 9):
            s = (ae == i) & (al == j) & np.isfinite(val)
            row.append("." if j >= i else ("-" if s.sum() < minn else fmt.format(agg(val[s]))))
        rows.append(row)
    return fmt_table(rows, ["age e \\ age l"] + [str(j) for j in range(0, 9)])


if __name__ == "__main__":
    # ---------------- runner-up vs third
    x = sample(2 ** 18, 1050)
    pM, pS, pT, M, S, Tv = top3(x)
    fourth = np.sort(x, 1)[:, -4]
    sel = (Tv > 0) & (M - S >= 0.35 * M) & (fourth < S - 0.2 * M) & (S > 0.3 * M)      # room for the sweep: x_l in [S-0.15M, S+0.15M]
    idx = np.flatnonzero(sel)[:60000]
    x, pM, pS, pT, M, S, Tv = (a[idx] for a in (x, pM, pS, pT, M, S, Tv))
    pe, pl = np.minimum(pS, pT), np.maximum(pS, pT)
    xs = x.copy(); r = np.arange(len(x)); xs[r, pe] = S                     # the earlier candidate has value S
    D, A = sweep(xs, pe, pl, "RT", M)
    eps, nch = crossing(D)
    print(f"# RT sweeps: n = {len(x)} sequences (seed 1050; T > 0, M - S >= 0.35 M, S > 0.3 M, fourth < S - 0.2 M); earlier candidate fixed at S, later one swept")
    summarize(eps, nch, "all")
    where = np.where(pM < pe, "M before both", np.where(pM > pl, "M after both", "M between"))
    for wname in ("M before both", "M between", "M after both"):
        summarize(eps[where == wname], nch[where == wname], f"  {wname}")
    summarize(eps[pl - pe == 1], nch[pl - pe == 1], "  adjacent (l = e + 1)")
    summarize(eps[pl == 9], nch[pl == 9], "  later candidate at position 9")
    summarize(eps[pl < 9], nch[pl < 9], "  later candidate before position 9")
    # what the net answers along the sweep
    i0 = np.searchsorted(EPS, 0.0)
    print("\n answers along the sweep (share): eps | answer = e | answer = l | answer = M | other | P(delta > 0)")
    for e in (-0.1, -0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
        i = int(np.argmin(np.abs(EPS - e)))
        a = A[:, i]
        print(f"   {EPS[i]:+.4f} | {np.mean(a == pe):.4f} | {np.mean(a == pl):.4f} | {np.mean(a == pM):.4f} | {np.mean((a != pe) & (a != pl) & (a != pM)):.4f} | {np.mean(D[:, i] > 0):.4f}")
    fin = np.isfinite(eps)
    ae, al, aM = 9 - pe, 9 - pl, 9 - pM
    print("\n## median eps* by ages (at the last step) of the earlier (rows) and later (columns) candidate; all M positions")
    print(age_table(eps, ae, al))
    print("\n## interquartile range of eps* by ages")
    print(age_table(eps, ae, al, fmt="{:.3f}", agg=lambda v: np.subtract(*np.quantile(v, [.75, .25]))))
    # how much of the variance of eps* is explained by the configuration (positions of e, l, M)?
    key = (pe * 100 + pl * 10 + pM)[fin]; v = eps[fin]
    uk, inv = np.unique(key, return_inverse=True)
    cm = np.bincount(inv, v) / np.bincount(inv)
    print(f"\n[FIT] share of var(eps*) explained by the cell (p_e, p_l, p_M) means: R^2 = {1 - ((v - cm[inv])**2).sum() / ((v - v.mean())**2).sum():.3f}  ({len(uk)} cells, sd(eps*) = {v.std():.4f})")
    key2 = (pe * 10 + pl)[fin]; uk2, inv2 = np.unique(key2, return_inverse=True); cm2 = np.bincount(inv2, v) / np.bincount(inv2)
    print(f"[FIT] by (p_e, p_l) only: R^2 = {1 - ((v - cm2[inv2])**2).sum() / ((v - v.mean())**2).sum():.3f}")
    np.savez_compressed(HIDE / "r1_I_errors" / "i04b_RT.npz", eps=eps, pe=pe, pl=pl, pM=pM, M=M, S=S, x=xs)

    # ---------------- max vs runner-up
    x = sample(2 ** 18, 1051)
    pM, pS, pT, M, S, Tv = top3(x)
    rng = np.random.default_rng(1052)
    sel = (M > 0) & (pM < 9)
    x, pM, M = x[sel], pM[sel], M[sel]
    pl = np.array([rng.integers(p + 1, 10) for p in pM])                   # a later position, uniform
    r = np.arange(len(x))
    xo = x.copy(); xo[r, pM] = -np.inf; xo[r, pl] = -np.inf
    keep = (xo.max(1) < 0.7 * M)
    idx = np.flatnonzero(keep)[:60000]
    x, pM, M, pl = x[idx], pM[idx], M[idx], pl[idx]
    D, A = sweep(x, pM, pl, "MR", M)
    eps, nch = crossing(D)
    print(f"\n# MR sweeps: n = {len(x)} sequences (seeds 1051/1052; all inputs other than the two < 0.7 M); max fixed at an earlier position, later input swept through it")
    summarize(eps, nch, "all")
    summarize(eps[pl - pM == 1], nch[pl - pM == 1], "  adjacent (l = e + 1)")
    summarize(eps[pl == 9], nch[pl == 9], "  later input at position 9")
    print("\n answers along the sweep (share): eps | answer = e (old max) | answer = l | other | P(delta > 0)")
    for e in (-0.1, -0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
        i = int(np.argmin(np.abs(EPS - e)))
        a = A[:, i]
        print(f"   {EPS[i]:+.4f} | {np.mean(a == pM):.4f} | {np.mean(a == pl):.4f} | {np.mean((a != pM) & (a != pl)):.4f} | {np.mean(D[:, i] > 0):.4f}")
    ae, al = 9 - pM, 9 - pl
    print("\n## median eps* (record offset) by ages of the earlier max (rows) and of the later input (columns)")
    print(age_table(eps, ae, al))
    print("\n## interquartile range of eps*")
    print(age_table(eps, ae, al, fmt="{:.3f}", agg=lambda v: np.subtract(*np.quantile(v, [.75, .25]))))
    # relation to the threshold of unit 4 at the step the later input arrives
    H = run(x)
    hprev = H[r[:len(x)], pl - 1]
    th4 = -(hprev @ W[4]) / w[4]
    sag = th4 / M - 1                                                      # negative = threshold below the max
    fin = np.isfinite(eps)
    print(f"\nthreshold of unit 4 when the later input arrives: theta_4/M - 1 quantiles 5/50/95%: " + " / ".join(f"{v:+.3f}" for v in np.quantile(sag, [.05, .5, .95])))
    print(f"P(eps* > theta_4/M - 1) = {np.mean(eps[fin] > sag[fin]):.4f}  (the record offset lies above the firing threshold of unit 4)")
    print(f"corr(eps*, theta_4/M - 1) = {np.corrcoef(eps[fin], sag[fin])[0, 1]:.3f}")
    print("median eps* and median sag by the gap between the two positions d = p_l - p_e (age of the max when the later input arrives):")
    for dd in range(1, 10):
        s = fin & (pl - pM == dd)
        print(f"   d = {dd}: n = {s.sum():5d}, median eps* = {np.median(eps[s]):+.4f}, IQR = {np.subtract(*np.quantile(eps[s], [.75, .25])):.4f}, median sag = {np.median(sag[s]):+.3f}, "
              f"P(eps* < 0) = {np.mean(eps[s] < 0):.3f}")
    np.savez_compressed(HIDE / "r1_I_errors" / "i04b_MR.npz", eps=eps, pe=pM, pl=pl, M=M, x=x, sag=sag)
