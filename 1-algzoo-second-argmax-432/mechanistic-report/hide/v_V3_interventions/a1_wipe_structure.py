"""Attack on I1, part 1: structure of the wipe result.  Seeds 5500..5503.

 1. What does the network answer in the cells where the wipe does NOT give the max?
 2. Does the result survive large later inputs (lead's selection, then no selection at all)?
 3. 'From then on': provisional answer (T3 readout) at every step after the wipe.
"""
from v3lib import *

out = Tee("a1_wipe_structure.txt")
WIPE = scale_units(COMP, 0.0)


def run_traj(x, tint, fn):
    """Like run_iv but returns all states (B, 10, 16) (state at tint is the intervened one)."""
    B = x.shape[0]
    h = np.zeros((B, N)); H = np.empty((B, T, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
        rows = np.flatnonzero(tint == t)
        if rows.size:
            h[rows] = fn(h[rows], rows)
        H[:, t] = h
    return H


# ---------------------------------------------------------------- 1. answer categories
x, pM, pS, M, S, nb = gather_I1(1_000_000, 5500)
out(f"[1] seeds 5500..{5500 + nb - 1}, I1-selected n = {len(x)}")
out("answer after the comparator wipe, by (k, t0): pM | true runner-up | position 9 (not runner-up) | "
    "other position > t0 | other position <= t0")
for k in range(1, 10):
    v = pM + k <= 9
    xv, pMv, pSv = x[v], pM[v], pS[v]
    t0 = pMv + k
    a = answer(run_iv(xv, t0, WIPE))
    for t in sorted(set(t0)):
        m = t0 == t
        am, pm, ps = a[m], pMv[m], pSv[m]
        c1 = am == pm; c2 = am == ps
        c3 = (am == 9) & ~c2 & ~c1
        c4 = (am > t) & ~c1 & ~c2 & ~c3
        c5 = ~(c1 | c2 | c3 | c4)
        if c1.mean() < 0.99:
            # among the misses: is the answer the largest input after t0 / the largest input before pM?
            xm = xv[m]
            later = np.where(np.arange(T)[None] > t, xm, -np.inf)
            la = later.argmax(1) if t < 9 else np.full(m.sum(), -1)
            out(f"  k={k} t0={t} n={m.sum():6d}: {c1.mean():.4f} | {c2.mean():.4f} | {c3.mean():.4f} | {c4.mean():.4f} | {c5.mean():.4f}"
                f"   P(ans = largest later input) = {np.mean(am == la):.4f}   mean(ans - pM | miss) = {np.mean((am - pm)[~c1]):+.2f}"
                f"   modal (ans - pM | miss) = {np.bincount((am - pm)[~c1] + 9).argmax() - 9:+d}")

# ---------------------------------------------------------------- 2. later inputs
out("\n[2a] lead's selection, wipe at t0 = pM + k <= 8: P(ans = pM) by where the true runner-up arrived and by")
out("     L/M, L = largest input after t0.  Columns: runner-up before t0 | runner-up after t0 | L/M <= 0 | (0,.5] | (.5,.8] | (.8,1)")
for k in range(1, 9):
    v = pM + k <= 8
    xv, pMv, pSv, Mv = x[v], pM[v], pS[v], M[v]
    t0 = pMv + k
    a = answer(run_iv(xv, t0, WIPE))
    hit = a == pMv
    L = np.where(np.arange(T)[None] > t0[:, None], xv, -np.inf).max(1) / Mv
    cells = [pSv <= t0, pSv > t0, L <= 0, (L > 0) & (L <= .5), (L > .5) & (L <= .8), L > .8]
    out(f"  k={k}: " + " | ".join(f"{hit[c].mean():.4f} (n={c.sum()})" for c in cells))

out("\n[2b] NO selection (all sequences, seeds 5510, 5511), wipe at t0 = pM + k <= 8: P(ans = pM) by gap (M - S)/M and by L/M")
xa = np.concatenate([sample(400_000, s) for s in (5510, 5511)])
oa = np.argsort(xa, axis=1)
pMa, pSa = oa[:, -1], oa[:, -2]
ra = np.arange(len(xa))
Ma, Sa = xa[ra, pMa], xa[ra, pSa]
base_a = answer(run(xa)[:, -1])
out(f"     unperturbed accuracy on all sequences: {np.mean(base_a == pSa):.4f}; P(ans = pM) = {np.mean(base_a == pMa):.4f}")
gapbins = [(-np.inf, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, np.inf)]
out("     columns: (M-S)/M in " + ", ".join(f"({a},{b}]" for a, b in gapbins) + "  | all with M > 0 | M <= 0?")
for k in range(1, 9):
    v = (pMa + k <= 8) & (Ma > 0)
    xv, pMv, pSv, Mv, Sv = xa[v], pMa[v], pSa[v], Ma[v], Sa[v]
    t0 = pMv + k
    a = answer(run_iv(xv, t0, WIPE))
    hit = a == pMv
    g = (Mv - Sv) / Mv
    out(f"  k={k}: " + " | ".join(f"{hit[(g > a_) & (g <= b_)].mean():.4f}" for a_, b_ in gapbins) + f"  | {hit.mean():.4f} (n={v.sum()})")
out("     same, P(ans = true runner-up):")
for k in range(1, 9):
    v = (pMa + k <= 8) & (Ma > 0)
    xv, pMv, pSv, Mv, Sv = xa[v], pMa[v], pSa[v], Ma[v], Sa[v]
    t0 = pMv + k
    a = answer(run_iv(xv, t0, WIPE))
    hit = a == pSv
    g = (Mv - Sv) / Mv
    out(f"  k={k}: " + " | ".join(f"{hit[(g > a_) & (g <= b_)].mean():.4f}" for a_, b_ in gapbins) + f"  | {hit.mean():.4f}")
out("     runner-up arrives AFTER the wipe and is within 5% of M ((M-S)/M <= 0.05, pS > t0), P(ans = pM), P(ans = pS):")
for k in range(1, 9):
    v = (pMa + k <= 8) & (Ma > 0)
    xv, pMv, pSv, Mv, Sv = xa[v], pMa[v], pSa[v], Ma[v], Sa[v]
    t0 = pMv + k
    a = answer(run_iv(xv, t0, WIPE))
    c = ((Mv - Sv) / Mv <= 0.05) & (pSv > t0)
    out(f"  k={k}: n={c.sum():6d}  P(pM) = {np.mean(a[c] == pMv[c]):.4f}  P(pS) = {np.mean(a[c] == pSv[c]):.4f}")

# ---------------------------------------------------------------- 3. provisional answer after the wipe
out("\n[3] provisional answer at step t (argmax of (U h_t)[i] over i >= 9 - t, position = i + t - 9), lead's selection.")
out("    P(provisional answer = pM) at t = t0, t0+1, ... after the wipe at t0 (pooled over t0 with t <= 9); '-' = no data")
H0 = run(x)
prov0 = np.stack([np.where(np.arange(T)[None] >= 9 - t, H0[:, t] @ U.T, -np.inf).argmax(1) + t - 9 for t in range(T)], 1)
rs = running_stats(x)
out("    unperturbed check of the T3 readout on these sequences: P(prov = second argmax of x_0..x_t), t = 1..9: "
    + " ".join(f"{np.mean(prov0[:, t] == rs['p2'][:, t]):.3f}" for t in range(1, 10)))
out("    unperturbed: P(prov = pM) at steps pM + j, j = 1..6: "
    + " ".join(f"{np.mean(prov0[(pM + j <= 9), :][np.arange((pM + j <= 9).sum()), (pM + j)[pM + j <= 9]] == pM[pM + j <= 9]):.4f}" for j in range(1, 7)))
for k in range(1, 9):
    v = pM + k <= 9
    xv, pMv = x[v], pM[v]
    t0 = pMv + k
    H = run_traj(xv, t0, WIPE)
    row = []
    for j in range(0, 9 - k + 1):
        m = t0 + j <= 9
        t = (t0 + j)[m]
        sc = np.einsum("bn,pn->bp", H[m, t], U)
        sc = np.where(np.arange(T)[None] >= (9 - t)[:, None], sc, -np.inf)
        prov = sc.argmax(1) + t - 9
        row.append(f"{np.mean(prov == pMv[m]):.4f}")
    out(f"  k={k}: steps after wipe j = 0.. : " + " ".join(row))
