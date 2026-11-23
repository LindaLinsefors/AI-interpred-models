"""V4. Claims 7 and 8: what each tape mode holds; the short clock.  Own code, own seeds, N(0,1).
A. Share of the variance of each tape mode's piece (indices 0-7, centred over the index, / M) explained by
   D3 = per (a_M, a_S): M, S;  D4 = per (a_M, a_S, a_T): M, S, T;  D5 = D4 + x_9, x_8.  Fit on half, score on half.
B. Functional at the final readout: one mode's piece / all four replaced by the description.
   Controls not run by the investigator:
     - the FITTED tape piece alone (no rest): if it ranks the runner-up first far more often than the true tape
       piece alone, the cell label a_S is doing the work
     - among the states where the network's top old entry is NOT the runner-up: does the substituted score
       reproduce the network's wrong choice?
C. Short clock: shares of the variance of z_short / M.
E. The test the investigator left open: substitute the description for the tape part of the STATE at t0 = 6 or 8
   and continue the run with the real inputs.  Share of final answers changed, against deletion and against
   donors (tape part of another state of the same cell, rescaled by M).
Run:  python -B v4_modes.py > v4_modes.txt"""
from vcommon import *

rng = np.random.default_rng(9400)
d = population(sample(500_000, 9401)); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; tape = sum(pc[m] for m in TAPE); rest = full - tape
topf = top_old(full, d["valid"]); train = ar < n // 2; test = ~train
M, S, T3 = d["M"], d["S"], d["T"]; x9, x8 = d["xr"][:, 9], d["xr"][:, 8]; Mn = M[:, None]
cMS = d["aM"] * 10 + d["aS"]; cMST = cMS * 10 + d["aT"]
DESCR = {"D2 a_M: M,S": ([M, S], d["aM"]), "D3 (a_M,a_S): M,S": ([M, S], cMS), "D4 (a_M,a_S,a_T): M,S,T": ([M, S, T3], cMST), "D5 D4 + x9,x8": ([M, S, T3, x9, x8], cMST)}
cen = lambda Sc: (Sc[:, :8] - Sc[:, :8].mean(1, keepdims=True)) / Mn
pS = lambda Sc, q=test: np.mean((top_old(Sc, d["valid"]) == d["iS"])[q])
ag = lambda Sc, q=test: np.mean((top_old(Sc, d["valid"]) == topf)[q])
print(f"{n} quiet final states (500000 sequences, seed 9401); {len(np.unique(cMS))} cells (a_M, a_S), {len(np.unique(cMST))} cells (a_M, a_S, a_T)")

print("\nA. share of variance of each mode's piece explained:   " + " | ".join(DESCR))
fits = {}
for m in TAPE + ["clock", "runner", "trans"]:
    Y = cen(pc[m]); tot = ((Y[test] - Y[test].mean(0)) ** 2).sum(); row = f"   {m:7s} rms {np.sqrt((Y[test] ** 2).mean()):6.2f}  "
    for dn, (F, cell) in DESCR.items():
        fits[m, dn] = cell_fit(pc[m] / Mn, np.stack(F, 1) / Mn, cell, train) * Mn
        row += f"  {1 - ((Y - cen(fits[m, dn]))[test] ** 2).sum() / tot:6.3f}"
    print(row)

print(f"\nB. functional at the readout (test half).  full {pS(full):.3f}; true tape alone {pS(tape):.3f}; tape deleted (rest alone) {pS(rest):.3f}")
wrong = test & (topf != d["iS"]); right = test & (topf == d["iS"])
for dn in DESCR:
    ft = sum(fits[m, dn] for m in TAPE); Sc = rest + ft
    one = "  ".join(f"{m} {pS(full - pc[m] + fits[m, dn]):.3f}" for m in TAPE)
    print(f"   {dn:26s} single mode replaced: {one} | all four: P(S top) {pS(Sc):.3f} agree {ag(Sc):.3f} | FITTED TAPE ALONE: {pS(ft):.3f} |"
          f" network right ({right.sum()}): substituted also right {pS(Sc, right):.3f} | network wrong ({wrong.sum()}): same wrong entry {ag(Sc, wrong):.3f}, runner-up on top {pS(Sc, wrong):.3f}")
print("   single modes deleted from the full score: " + "  ".join(f"{m} {pS(full - pc[m]):.3f}" for m in TAPE) + "   single mode alone: " + "  ".join(f"{m} {pS(pc[m]):.3f}" for m in TAPE))

# ---------------------------------------------------------------- C. short clock
ho = d["h"][:, ORD]; zs = (ho @ LVEC["short"]) / M; zc = (ho @ LVEC["clock"]) / M
tot = (np.abs(zs[test] - zs[test].mean()) ** 2).sum()
share = lambda pred: 1 - (np.abs(zs - pred)[test] ** 2).sum() / tot
def cfit(F, cell):
    p = cell_fit(np.stack([zs.real, zs.imag], 1), np.stack(F, 1) / Mn, cell, train); return p[:, 0] + 1j * p[:, 1]
k = (np.conj(zc[train]) * zs[train]).sum() / (np.abs(zc[train]) ** 2).sum()
pk = np.zeros(n, complex)
for a in range(2, 10):
    q = d["aM"] == a; kk = (np.conj(zc[q & train]) * zs[q & train]).sum() / (np.abs(zc[q & train]) ** 2).sum(); pk[q] = kk * zc[q]
print("\nC. short clock, shares of sum |z - mean|^2 (z = coordinate / M; the share does not depend on the gauge of the eigenvector):")
print(f"   one kappa x clock coordinate {share(k * zc):.3f};  kappa per a_M {share(pk):.3f};  alpha(a_M)(M - S) {share(cfit([M - S], d['aM'])):.3f};  per a_M: M, S {share(cfit([M, S], d['aM'])):.3f};"
      f"  per a_S: M, S {share(cfit([M, S], d['aS'])):.3f};  per (a_M, a_S): M, S {share(cfit([M, S], cMS)):.3f};  top three {share(cfit([M, S, T3], cMST)):.3f};  + x9, x8 {share(cfit([M, S, T3, x9, x8], cMST)):.3f}")
print(f"   attack: affine in the clock coordinate with a free intercept per a_M (z = k z_clock + c0): {share(cfit([M, zc.real * M, zc.imag * M], d['aM'])):.3f};"
      f"  the same per (a_M, a_S): {share(cfit([M, zc.real * M, zc.imag * M], cMS)):.3f}")

# ---------------------------------------------------------------- E. substitution in the state, run continued
print("\nE. description substituted for the tape part of the STATE at t0, run continued with the real inputs (fit on half the quiet states, numbers from the other half)")
B = 400_000
x = sample(B, 9402); H = run(x); o9 = np.argsort(-x, 1); pM9, pS9 = o9[:, 0], o9[:, 1]
for t0 in (6, 8):
    st = stats(x, t0); q = np.flatnonzero(st["quiet"]); nq = len(q); tr = np.arange(nq) < nq // 2; te = ~tr
    h = H[q, t0]; fut = x[q, t0 + 1:]; Mq, Sq, Tq = st["M"][q], st["S"][q], st["T"][q]
    c2 = st["aM"][q] * 10 + st["aS"][q]; c3 = c2 * 10 + st["aT"][q]
    ans = lambda hh: (run(fut, h0=hh)[:, -1] @ U.T).argmax(1)
    a0 = ans(h); tp = h[:, ORD] @ P_TAPE.T; base = h.copy(); base[:, ORD] -= tp
    cls = {"all quiet": np.ones(nq, bool), "same top two": (pM9[q] == st["pM"][q]) & (pS9[q] == st["pS"][q]), "max overtaken": (pM9[q] > t0) & (pS9[q] == st["pM"][q]),
           "new runner-up": (pM9[q] == st["pM"][q]) & (pS9[q] > t0)}
    def put(part):
        out = base.copy(); out[:, ORD] += part; return out
    def donors(cell):
        out = np.arange(nq); order = np.argsort(cell, kind="stable"); bounds = np.flatnonzero(np.diff(cell[order])) + 1
        for idx in np.split(order, bounds):
            t_ = idx[tr[idx]]
            if len(t_):
                out[idx] = rng.choice(t_, len(idx))
        return out
    variants = {"tape deleted": base}
    for nm, F, cell in (("fit per (a_M,a_S): M,S", [Mq, Sq], c2), ("fit per (a_M,a_S,a_T): M,S,T", [Mq, Sq, Tq], c3)):
        variants[nm] = put(cell_fit(tp / Mq[:, None], np.stack(F, 1) / Mq[:, None], cell, tr) * Mq[:, None])
    for nm, cell in (("donor: any quiet state", np.zeros(nq, int)), ("donor: same (a_M,a_S)", c2), ("donor: same (a_M,a_S,a_T)", c3)):
        dn = donors(cell); variants[nm] = put(tp[dn] * (Mq / Mq[dn])[:, None])
    # donor with the same (a_M, a_S, a_T) and the nearest S/M (neighbour in the sorted order within the cell)
    order = np.lexsort((Sq / Mq, c3)); nb = np.empty(nq, int); nb[order] = np.roll(order, 1); okn = c3[nb] == c3
    variants["donor: same cell, nearest S/M"] = put(tp[nb] * (Mq / Mq[nb])[:, None])
    print(f"   t0 = {t0}: {nq} quiet states; share of final answers changed (clipped at zero in brackets; neg = share with a negative activation)")
    print("      variant                          " + "  ".join(f"{c:>22s}" for c in cls))
    for nm, hv in variants.items():
        a1 = ans(hv); a1c = ans(np.maximum(hv, 0)); sel = te & okn if "nearest" in nm else te
        print(f"      {nm:32s} " + "  ".join(f"{np.mean((a1 != a0)[sel & c]):.3f} ({np.mean((a1c != a0)[sel & c]):.3f}) n{(sel & c).sum():6d}" for c in cls.values())
              + f"   neg {np.mean((hv < -1e-9).any(1)[sel]):.3f}")
