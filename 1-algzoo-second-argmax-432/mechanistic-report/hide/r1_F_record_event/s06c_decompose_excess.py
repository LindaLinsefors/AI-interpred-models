"""s06c (tasks F3 iii, F5, F6). Exact decomposition of the final logits at a record into
'what existed before' and 'what the record created', with the record written as five threshold excesses.

Sequences: record at s = 3..9, M0 > 0, x_s final max, old max p0 = true final runner-up, S0 > 0 (seeds 1040, 1041).
No-record counterfactual 'cf': only x_s is replaced by x_s' = min(S0,0) - 1.

At the record step the pre-activations are  z_s = W h_{s-1} + w x_s.  A clipped comparator j has z_s[j] < 0, and
    h_s = h_s^lin + sum_j c_j e_j + h4(s) e_4 ,       c_j = -z_s[j] = |w_j| (x_s - theta_j(s)) > 0,
where h_s^lin is the state with the comparators NOT clipped (z_s[j] kept, negative) and unit 4 off: the state the
resting (linear) flow would have produced.  With P = D_9 W ... D_{s+1} W (actual gates after s):

   logits_act - logits_cf  =  sum_j c_j U P e_j          'comparator excesses'   C
                            +  h4(s) U P e_4             'unit-4 pulse'          P4
                            +  [U P h_s^lin + later inputs]  -  logits_cf        'other gate changes' G (linear flow read
                                                                                  through the actual instead of the cf gates)
c_j is split exactly:  c_j = |w_j| x_s - W[j].h_{s-1},  and W[j].h_{s-1} by mode of h_{s-1}.  The clock part,
-W[j,ORD].2Re(z R_clock), is exactly 'the comparator half of the (propagated) clock mode, removed'.
All logits are relative to the mean logit of the bystanders (positions < s other than p0, q0).
"""
from fcommon import *

PARTS = NAMES + ["pulses"]


def phi_from(G, t0):
    B = G.shape[0]
    M = np.broadcast_to(np.eye(16), (B, 16, 16)).copy()
    for t in range(t0 + 1, 10):
        M = np.einsum("ij,bjk->bik", W, M) * G[:, t, :, None]
    return M


def mode_vectors(h):
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
        n = len(i); ar = np.arange(n)
        xa = x[i]
        xc = xa.copy(); xc[:, s] = np.minimum(q["S0"][i], 0) - 1
        Ha, Za = run(xa, return_pre=True); Ga = Za > 0
        Hc, Zc = run(xc, return_pre=True); Gc = Zc > 0
        la, lc = Ha[:, 9] @ U.T, Hc[:, 9] @ U.T
        hprev, hs = Ha[:, s - 1], Ha[:, s]
        P = phi_from(Ga, s)                                   # h_s -> h_9
        UP = np.einsum("qi,bij->bqj", U, P)                   # (n, 10, 16): logit kernel of each unit of h_s
        if s <= 8:                                            # the part of UP that passes through unit 5 / unit 9 at s+1
            P1 = phi_from(Ga, s + 1)
            UP1 = np.einsum("qi,bij->bqj", U, P1)
            D1 = Ga[:, s + 1].astype(float)
            via = {pu: UP1[:, :, pu, None] * (W[pu][None, None, :] * D1[:, pu, None, None]) for pu in (5, 9)}   # (n,10,16)
        else:
            via = {pu: np.zeros_like(UP) for pu in (5, 9)}
        off = ~Ga[:, s][:, COMP]                              # (n,4) comparators clipped at s
        mv = mode_vectors(hprev)

        def kick(coef, K=UP):
            """coef (n,4) per comparator -> logits (n,10)"""
            return np.einsum("bqj,bj->bq", K[:, :, COMP], coef * off)

        C_x = kick(np.abs(w[COMP])[None, :] * xa[:, s, None])
        C_k = {k: kick(-(v @ W[COMP].T)) for k, v in mv.items()}
        C_all = C_x + sum(C_k.values())
        P4 = UP[:, :, 4] * hs[:, 4, None]
        C_via = {pu: kick(np.abs(w[COMP])[None, :] * xa[:, s, None] - hprev @ W[COMP].T, via[pu]) for pu in (5, 9)}
        P4_via = {pu: via[pu][:, :, 4] * hs[:, 4, None] for pu in (5, 9)}
        Cclock_via = {pu: kick(-(mv["clock"] @ W[COMP].T), via[pu]) for pu in (5, 9)}
        G_tot = la - lc - C_all - P4
        # old state through D_lin and actual later gates, by mode, minus through cf gates
        Dlin = Ga[:, s].astype(float).copy(); Dlin[:, COMP] = 1; Dlin[:, 4] = 0
        Pc = phi_from(Gc, s - 1)
        lin_old = {k: np.einsum("bqj,bj->bq", UP, (v @ W.T) * Dlin) for k, v in mv.items()}
        old_cf = {k: np.einsum("bij,bj->bi", Pc, v) @ U.T for k, v in mv.items()}
        G_old = {k: lin_old[k] - old_cf[k] for k in PARTS}
        G_in = G_tot - sum(G_old.values())

        p0, q0 = q["p0"][i], q["q0"][i]
        by = np.ones((n, 10), bool); by[:, s:] = False; by[ar, p0] = False; by[ar, q0] = False
        nby = by.sum(1)
        late = np.zeros((n, 10), bool); late[:, s + 1:] = True

        def ref(v):
            b = (v * by).sum(1) / np.maximum(nby, 1)
            lt = (v * late).sum(1) / np.maximum(late.sum(1), 1)
            return np.column_stack([v[ar, p0] - b, v[ar, q0] - b, v[ar, s] - b, lt - b])

        keep = nby >= 1
        th = -(hprev @ W[COMP + [4]].T) / w[COMP + [4]]       # thresholds theta_j(s), j = 1,6,7,8,4
        for nm, v in (("s", np.full(n, s)), ("g", q["g"][i]), ("Mn", q["Mn"][i]), ("M0", q["M0"][i]), ("S0", q["S0"][i]),
                      ("theta", th), ("alloff", off.all(1)), ("ok", la.argmax(1) == p0),
                      ("act", ref(la)), ("cf", ref(lc)), ("C_x", ref(C_x)), ("C_all", ref(C_all)), ("P4", ref(P4)),
                      ("G_tot", ref(G_tot)), ("G_in", ref(G_in)), ("C_via5", ref(C_via[5])), ("C_via9", ref(C_via[9])),
                      ("P4_via5", ref(P4_via[5])), ("P4_via9", ref(P4_via[9])),
                      ("Cclock_via5", ref(Cclock_via[5])), ("Cclock_via9", ref(Cclock_via[9]))):
            put(nm, v[keep])
        for k in PARTS:
            put("C_" + k, ref(C_k[k])[keep]); put("G_" + k, ref(G_old[k])[keep])

R = {k: np.concatenate(v) for k, v in acc.items()}
# compact cache for s10_checks.py (only the columns it needs, float32)
np.savez_compressed(HIDE / "r1_F_record_event" / "s06c_terms.npz",
                    **{k: R[k].astype(np.float32) for k in ("s", "g", "act", "cf", "C_clock", "Mn", "M0", "S0")})
R["C_nonclock"] = R["C_all"] - R["C_clock"]
R["created"] = R["act"] - R["cf"]
R["G_oldsum"] = sum(R["G_" + k] for k in PARTS)
n = len(R["s"])
COLS = ["p0 (old max)", "q0 (old 2nd)", "s (new max)", "later positions"]
print(f"sequences: n = {n}; network answers p0 in {R['ok'].mean():.4f}; all four comparators clipped at s in {R['alloff'].mean():.4f}")
ROWS = [("cf", "EXISTED: no-record counterfactual logits"),
        ("created", "CREATED by the record (actual - counterfactual)"),
        ("C_clock", "  C  comparator excess, clock part"),
        ("C_nonclock", "  C  comparator excess, all other parts"),
        ("P4", "  P4 unit-4 pulse"),
        ("G_tot", "  G  other gate changes (linear flow re-read)"),
        ("act", "ACTUAL logits"),
        ("C_x", "    [C: |w_j| x_s part]"),
        ("C_level", "    [C: level part of the thresholds]"),
        ("C_runner", "    [C: runner part of the thresholds]"),
        ("C_rest", "    [C: h2, alt, h3, short, trans, echo, pulses]"),
        ("Cclock_via5", "    [C clock part: of which via unit 5 at s+1]"),
        ("Cclock_via9", "    [C clock part: of which via unit 9 at s+1]"),
        ("C_via5", "    [C total: of which via unit 5 at s+1]"),
        ("C_via9", "    [C total: of which via unit 9 at s+1]"),
        ("P4_via5", "    [P4: of which via unit 5 at s+1]"),
        ("P4_via9", "    [P4: of which via unit 9 at s+1]"),
        ("G_clock", "    [G: clock component of the old state]"),
        ("G_level", "    [G: level component]"),
        ("G_runner", "    [G: runner component]"),
        ("G_restold", "    [G: other components of the old state]"),
        ("G_in", "    [G: inputs x_s, x_(s+1).. vs cf inputs]")]
R["C_rest"] = sum(R["C_" + k] for k in PARTS if k not in ("level", "runner", "clock"))
R["G_restold"] = sum(R["G_" + k] for k in PARTS if k not in ("level", "runner", "clock"))


def table(mask, title):
    print(f"\n=== {title}: n = {mask.sum()} ===   mean logit relative to bystanders")
    print(f"  {'term':52s}" + "".join(f"{c:>18s}" for c in COLS))
    for name, lab in ROWS:
        print(f"  {lab:52s}" + "".join(f"{R[name][mask, j].mean():18.2f}" for j in range(4)))


g2 = R["g"] >= 2
table(g2, "all s = 3..9, g >= 2")
table(g2 & (R["s"] <= 7), "s = 3..7, g >= 2")
table(g2 & (R["s"] == 8), "s = 8, g >= 2")
table(g2 & (R["s"] == 9), "s = 9, g >= 2")
table(R["g"] == 1, "all s = 3..9, g = 1")
table((R["g"] == 1) & (R["s"] == 9), "s = 9, g = 1")

X = [R["M0"] - R["S0"], R["S0"], R["Mn"] - R["M0"]]
print("\n=== regressions (g >= 2), bystander-referenced: term ~ a (M0-S0) + b S0 + c (Mn-M0)  [no intercept] ===")
for j, c in ((0, "p0"), (1, "q0"), (2, "s ")):
    for name, lab in ROWS[:7]:
        b, r2 = ols(R[name][g2, j], [v[g2] for v in X])
        print(f"  logit[{c}] {lab:50s}: {b[0]:8.2f} (M0-S0) {b[1]:+8.2f} S0 {b[2]:+8.2f} (Mn-M0)   R2 = {r2:.3f}   mean {R[name][g2, j].mean():7.2f}")
    print()
print("margin logit[p0] - logit[q0]:")
for name, lab in ROWS[:7]:
    y = R[name][g2, 0] - R[name][g2, 1]
    b, r2 = ols(y, [v[g2] for v in X])
    print(f"  {lab:50s}: {b[0]:8.2f} (M0-S0) {b[1]:+8.2f} S0 {b[2]:+8.2f} (Mn-M0)   R2 = {r2:.3f}   mean {y.mean():7.2f}")
print("margin logit[p0] - logit[s]:")
for name, lab in ROWS[:7]:
    y = R[name][g2, 0] - R[name][g2, 2]
    b, r2 = ols(y, [v[g2] for v in X])
    print(f"  {lab:50s}: {b[0]:8.2f} (M0-S0) {b[1]:+8.2f} S0 {b[2]:+8.2f} (Mn-M0)   R2 = {r2:.3f}   mean {y.mean():7.2f}")

print("\n=== comparator and unit-4 thresholds at the record step, by age g of the old max:  M0 - theta_j(s) ~ a (M0-S0) + b S0 ===")
print("   g      n  |  " + "  ".join(f"unit {u}: a      b     R2 " for u in COMP + [4]))
for g in range(1, 8):
    m = R["g"] == g
    row = f"   {g} {m.sum():6d}  | "
    for c in range(5):
        y = R["M0"][m] - R["theta"][m, c]
        b, r2 = ols(y, [R["M0"][m] - R["S0"][m], R["S0"][m]])
        row += f"     {b[0]:6.3f} {b[1]:6.3f} {r2:5.2f}  "
    print(row)

print("\n=== created logit at p0, by s (g >= 2): mean of each term ===")
print("   s      n   existed   created = C_clock + C_nonclock +   P4   +   G     actual | P(ans = p0)")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    print(f"   {s} {m.sum():6d}  {R['cf'][m, 0].mean():7.2f}   {R['created'][m, 0].mean():7.2f}   {R['C_clock'][m, 0].mean():7.2f}   {R['C_nonclock'][m, 0].mean():7.2f}"
          f"   {R['P4'][m, 0].mean():7.2f}  {R['G_tot'][m, 0].mean():7.2f}   {R['act'][m, 0].mean():7.2f} | {R['ok'][m].mean():.3f}")
print("   same at q0:")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    print(f"   {s} {m.sum():6d}  {R['cf'][m, 1].mean():7.2f}   {R['created'][m, 1].mean():7.2f}   {R['C_clock'][m, 1].mean():7.2f}   {R['C_nonclock'][m, 1].mean():7.2f}"
          f"   {R['P4'][m, 1].mean():7.2f}  {R['G_tot'][m, 1].mean():7.2f}   {R['act'][m, 1].mean():7.2f}")
print("   same at s (the new max):")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    print(f"   {s} {m.sum():6d}  {R['cf'][m, 2].mean():7.2f}   {R['created'][m, 2].mean():7.2f}   {R['C_clock'][m, 2].mean():7.2f}   {R['C_nonclock'][m, 2].mean():7.2f}"
          f"   {R['P4'][m, 2].mean():7.2f}  {R['G_tot'][m, 2].mean():7.2f}   {R['act'][m, 2].mean():7.2f}")
