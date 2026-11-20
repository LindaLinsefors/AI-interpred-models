"""s06 (tasks F3 iii and F5). Exact decomposition of the final logits at a record.

Sequences: record at s (3..9), M0 > 0, x_s is the final max, the old max (p0) is the true final runner-up,
the old runner-up q0 has S0 > 0.  N(0,1), seed 1040 (400k) + 1041 (400k).

With the gates of a run held fixed the net is linear:   h_9 = Phi(9 <- s-1) h_{s-1} + sum_{t>=s} (...) x_t,
Phi = D_9 W D_8 W ... D_s W.   Two runs: the actual one (gates 'act') and a no-record counterfactual 'cf'
in which only x_s is replaced by a low value x_s' = min(S0, 0) - 1.

   logits_act = U Phi_cf h_{s-1}                 (1) OLD STATE AS IT WOULD HAVE BEEN READ WITHOUT THE RECORD
              + U (Phi_act - Phi_cf) h_{s-1}     (2) OLD STATE, RE-READ BECAUSE THE RECORD CHANGED THE GATES ('conversion')
              + N_act                            (3) NEW INPUTS x_s..x_9 through the actual gates
   logits_cf  = U Phi_cf h_{s-1} + N_cf          (the no-record logits)
(2) is split exactly by mode of h_{s-1} (h_{s-1}[ORD] = sum_k m_k RIGHT_k, plus the pulse entries of h_{s-1}).
All logits are reported relative to the mean logit of the 'bystanders' (positions < s other than p0, q0).
"""
from fcommon import *

PARTS = NAMES + ["pulses"]


def phi(G, s):
    """G (B, 10, 16) boolean gates -> Phi (B,16,16) = D_9 W ... D_s W."""
    B = G.shape[0]
    M = np.broadcast_to(np.eye(16), (B, 16, 16)).copy()
    for t in range(s, 10):
        M = np.einsum("ij,bjk->bik", W, M) * G[:, t, :, None]
    return M


def mode_vectors(h):
    """h (B,16) -> dict part -> (B,16) vectors summing to h."""
    out = {}
    ho = h[:, ORD]
    for n in NAMES:
        z = ho @ LEFT[n]
        v = np.zeros_like(h)
        v[:, ORD] = 2 * np.real(np.multiply.outer(z, RIGHT[n])) if IS_COMPLEX[n] else np.real(np.multiply.outer(z, RIGHT[n]))
        out[n] = v
    v = np.zeros_like(h); v[:, PULSE] = h[:, PULSE]
    out["pulses"] = v
    return out


acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)

for seed in (1040, 1041):
    x = sample(400_000, seed)
    st = running_stats(x)
    order = np.argsort(-x, axis=1)
    for s in range(3, 10):
        rec, q = records(x, s, st)
        ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"]) & (q["S0"] > 0)
        i = np.flatnonzero(ok)
        n = len(i)
        ar = np.arange(n)
        xa = x[i]
        xc = xa.copy(); xc[:, s] = np.minimum(q["S0"][i], 0) - 1
        Ha, Za = run(xa, return_pre=True)
        Hc, Zc = run(xc, return_pre=True)
        Pa, Pc = phi(Za > 0, s), phi(Zc > 0, s)
        hprev = Ha[:, s - 1]
        la, lc = Ha[:, 9] @ U.T, Hc[:, 9] @ U.T
        old_cf = np.einsum("bij,bj->bi", Pc, hprev) @ U.T
        old_act = np.einsum("bij,bj->bi", Pa, hprev) @ U.T
        conv = old_act - old_cf
        N_act, N_cf = la - old_act, lc - old_cf
        mv = mode_vectors(hprev)
        conv_k = {k: (np.einsum("bij,bj->bi", Pa - Pc, v) @ U.T) for k, v in mv.items()}
        oldcf_k = {k: (np.einsum("bij,bj->bi", Pc, v) @ U.T) for k, v in mv.items()}
        assert np.allclose(sum(conv_k.values()), conv, atol=1e-6)
        p0, q0 = q["p0"][i], q["q0"][i]
        by = np.ones((n, 10), bool)
        by[:, s:] = False; by[ar, p0] = False; by[ar, q0] = False
        nby = by.sum(1)
        late = np.zeros((n, 10), bool); late[:, s + 1:] = True

        def ref(v):
            """(n,10) logits -> (n,4): values at p0, q0, s, mean later; all minus the bystander mean."""
            b = (v * by).sum(1) / np.maximum(nby, 1)
            lt = (v * late).sum(1) / np.maximum(late.sum(1), 1)
            return np.column_stack([v[ar, p0] - b, v[ar, q0] - b, v[ar, s] - b, lt - b])

        keep = nby >= 1
        put("s", np.full(n, s)[keep]); put("g", q["g"][i][keep])
        put("Mn", q["Mn"][i][keep]); put("M0", q["M0"][i][keep]); put("S0", q["S0"][i][keep])
        for name, v in (("act", la), ("cf", lc), ("old_cf", old_cf), ("conv", conv), ("N_act", N_act), ("N_cf", N_cf)):
            put(name, ref(v)[keep])
        for k in PARTS:
            put("conv_" + k, ref(conv_k[k])[keep])
            put("oldcf_" + k, ref(oldcf_k[k])[keep])
        put("ans_act_p0", (la.argmax(1) == p0)[keep]); put("ans_cf_q0", (lc.argmax(1) == q0)[keep])
        put("ans_cf_p0", (lc.argmax(1) == p0)[keep])
        # sensitivity of the counterfactual to the choice of x_s'
        xc2 = xc.copy(); xc2[:, s] -= 1.0
        lc2 = run(xc2)[:, 9] @ U.T
        put("cf_sens", (ref(lc2) - ref(lc))[keep])
        put("cf_rest", ((Zc[:, s][:, COMP] > 0).all(1) & ~(Zc[:, s][:, PULSE] > 0)[:, 0])[keep])

R = {k: np.concatenate(v) for k, v in acc.items()}
n = len(R["s"])
COLS = ["p0 (old max)", "q0 (old 2nd)", "s (new max)", "later positions"]
print(f"sequences: n = {n}; network answers p0 in {R['ans_act_p0'].mean():.4f}; the no-record counterfactual answers q0 in "
      f"{R['ans_cf_q0'].mean():.4f} and p0 in {R['ans_cf_p0'].mean():.4f}; cf step is a resting step (comparators on, unit 4 off) in {R['cf_rest'].mean():.4f}")
print(f"sensitivity of the counterfactual logits to lowering x_s' by another 1.0 (mean |change|, bystander-referenced): "
      + ", ".join(f"{c}: {np.abs(R['cf_sens'][:, j]).mean():.2f}" for j, c in enumerate(COLS)))


def table(mask, title):
    print(f"\n=== {title}: n = {mask.sum()} ===   mean logit relative to bystanders (positions < s other than p0, q0)")
    print(f"  {'term':46s}" + "".join(f"{c:>18s}" for c in COLS))
    for name, lab in (("cf", "no-record counterfactual, total"),
                      ("old_cf", "(1) old state, read without the record"),
                      ("conv", "(2) old state, conversion by the record's gates"),
                      ("N_act", "(3) new inputs x_s..x_9"),
                      ("act", "actual logits = (1)+(2)+(3)"),
                      ("N_cf", "    [new inputs in the counterfactual]")):
        print(f"  {lab:46s}" + "".join(f"{R[name][mask, j].mean():18.2f}" for j in range(4)))
    print("  split of (2) by mode of h_(s-1):")
    for k in PARTS:
        print(f"     {k:43s}" + "".join(f"{R['conv_' + k][mask, j].mean():18.2f}" for j in range(4)))
    print("  split of (1) by mode of h_(s-1):")
    for k in PARTS:
        print(f"     {k:43s}" + "".join(f"{R['oldcf_' + k][mask, j].mean():18.2f}" for j in range(4)))


g2 = R["g"] >= 2
table(g2, "all s = 3..9, g >= 2")
table(R["g"] == 1, "all s = 3..9, g = 1")
for s in (4, 6, 8, 9):
    table(g2 & (R["s"] == s), f"s = {s}, g >= 2")

print("\n=== regressions (g >= 2): term ~ a (M0 - S0) + b S0 + c (Mn - M0), no intercept ===")
X = [R["M0"] - R["S0"], R["S0"], R["Mn"] - R["M0"]]
for j, c in ((0, "p0"), (1, "q0"), (2, "s")):
    for name in ("cf", "old_cf", "conv", "conv_clock", "conv_level", "conv_runner", "N_act", "act"):
        b, r2 = ols(R[name][g2, j], [v[g2] for v in X])
        print(f"  logit[{c}] - bystanders, {name:11s}: {b[0]:8.2f} (M0-S0) {b[1]:+8.2f} S0 {b[2]:+8.2f} (Mn-M0)    R2 = {r2:.3f}")
    print()
print("margin logit[p0] - logit[q0] by term (g >= 2):")
for name in ("cf", "old_cf", "conv", "conv_clock", "N_act", "act"):
    y = R[name][g2, 0] - R[name][g2, 1]
    b, r2 = ols(y, [v[g2] for v in X])
    print(f"  {name:11s}: mean {y.mean():7.2f}   ~ {b[0]:8.2f} (M0-S0) {b[1]:+8.2f} S0 {b[2]:+8.2f} (Mn-M0)    R2 = {r2:.3f}")
print("\nby s (g >= 2): coefficient of (M0-S0) in conv_clock at p0, and in the margin p0-q0 of the actual logits")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    b1, r1 = ols(R["conv_clock"][m, 0], [v[m] for v in X])
    b2, r2 = ols(R["act"][m, 0] - R["act"][m, 1], [v[m] for v in X])
    b3, r3 = ols(R["conv"][m, 0], [v[m] for v in X])
    print(f"  s={s}: conv_clock[p0] ~ {b1[0]:6.1f} (M0-S0) {b1[1]:+6.1f} S0 {b1[2]:+6.1f} (Mn-M0) (R2 {r1:.2f}) | conv[p0] ~ {b3[0]:6.1f} {b3[1]:+6.1f} {b3[2]:+6.1f} (R2 {r3:.2f})"
          f" | actual margin p0-q0 ~ {b2[0]:6.1f} (M0-S0) {b2[1]:+6.1f} S0 {b2[2]:+6.1f} (Mn-M0) (R2 {r2:.2f})")

print("\n=== F5: what the record does to the old runner-up's logit (q0), relative to bystanders (g >= 2) ===")
d = R["act"][g2, 1] - R["cf"][g2, 1]
print(f"  mean logit[q0]-bystanders: no record {R['cf'][g2, 1].mean():.2f} -> with record {R['act'][g2, 1].mean():.2f}  (change {d.mean():.2f}, sd {d.std():.2f})")
d0 = R["act"][g2, 0] - R["cf"][g2, 0]
print(f"  mean logit[p0]-bystanders: no record {R['cf'][g2, 0].mean():.2f} -> with record {R['act'][g2, 0].mean():.2f}  (change {d0.mean():.2f}, sd {d0.std():.2f})")
b, r2 = ols(d, [v[g2] for v in X]); print(f"  change at q0 ~ {b[0]:.2f} (M0-S0) {b[1]:+.2f} S0 {b[2]:+.2f} (Mn-M0)   R2 = {r2:.3f}")
b, r2 = ols(d0, [v[g2] for v in X]); print(f"  change at p0 ~ {b[0]:.2f} (M0-S0) {b[1]:+.2f} S0 {b[2]:+.2f} (Mn-M0)   R2 = {r2:.3f}")
print("  would q0 still beat p0 if only p0's logit had changed? i.e. is act[p0] > cf[q0] (both vs bystanders):",
      f"{(R['act'][g2, 0] > R['cf'][g2, 1]).mean():.3f};  actual: act[p0] > act[q0]: {(R['act'][g2, 0] > R['act'][g2, 1]).mean():.3f}")
print("  by s: mean change at q0 | mean change at p0 | mean no-record logit of q0 | P(act[p0] > cf[q0])")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    print(f"    s={s}: {(R['act'][m, 1] - R['cf'][m, 1]).mean():7.2f} | {(R['act'][m, 0] - R['cf'][m, 0]).mean():7.2f} | {R['cf'][m, 1].mean():7.2f} | {(R['act'][m, 0] > R['cf'][m, 1]).mean():.3f}")
np.savez(HIDE / "r1_F_record_event" / "s06_terms.npz", **{k: v for k, v in R.items()})
