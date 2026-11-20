"""s06b (tasks F3 iii, F5, F6). Finer exact path decomposition of the final logits at a record.

Same sequences as s06 (record at s = 3..9, M0 > 0, x_s final max, old max p0 = true final runner-up, S0 > 0;
seeds 1040, 1041) and the same no-record counterfactual (x_s' = min(S0,0) - 1).

The state at the record step is split exactly as
    h_s = h4(s) e_4                         'record pulse'  (h4(s) = w4 (x_s - theta4(s)), kept as ONE term)
        + sum_k D'_s W [mode k of h_{s-1}]  old state after the gates of step s (D'_s = actual gates, unit 4 removed)
        + D'_s w x_s                        direct input into the units that are on (only wave units: comparators are off)
and each piece is propagated to the end through the actual gates, split by whether the path passes through
unit 5 at s+1, unit 9 at s+1, or neither ('direct').  Later inputs x_{s+1..9} are the remainder.
Counterfactual:  old state read with the no-record gates, U Phi_cf h_{s-1}, by mode.

   conversion of mode k  :=  [old state mode k, direct paths, actual gates]  -  [old state mode k, no-record gates]
All logits are relative to the mean logit of the bystanders (positions < s other than p0, q0).
"""
from fcommon import *

PARTS = NAMES + ["pulses"]


def phi_from(G, t0):
    """Maps h_{t0} -> its contribution to h_9:  D_9 W ... D_{t0+1} W  (identity if t0 = 9).  (B,16,16)."""
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
        Pc = phi_from(Gc, s - 1)
        Ds = Ga[:, s].astype(float).copy(); Ds[:, 4] = 0          # D'_s
        if s <= 8:
            P1 = phi_from(Ga, s + 1)
            D1 = Ga[:, s + 1].astype(float)

        def paths(v):
            """source v at step s (n,16) -> logits via direct, via unit 5 at s+1, via unit 9 at s+1."""
            if s == 9:
                return v @ U.T, np.zeros((n, 10)), np.zeros((n, 10))
            u = (v @ W.T) * D1
            u5 = np.zeros_like(u); u5[:, 5] = u[:, 5]
            u9 = np.zeros_like(u); u9[:, 9] = u[:, 9]
            ud = u - u5 - u9
            f = lambda a: np.einsum("bij,bj->bi", P1, a) @ U.T
            return f(ud), f(u5), f(u9)

        mv = mode_vectors(hprev)
        v4 = np.zeros((n, 16)); v4[:, 4] = hs[:, 4]
        vx = Ds * w * xa[:, s, None]
        t4 = paths(v4); tx = paths(vx)
        told = {k: paths((v @ W.T) * Ds) for k, v in mv.items()}
        oldcf = {k: np.einsum("bij,bj->bi", Pc, v) @ U.T for k, v in mv.items()}
        tot_s = sum(t4) + sum(tx) + sum(sum(t) for t in told.values())
        later = la - tot_s
        if s == 9:
            assert np.abs(later).max() < 1e-8

        p0, q0 = q["p0"][i], q["q0"][i]
        by = np.ones((n, 10), bool); by[:, s:] = False; by[ar, p0] = False; by[ar, q0] = False
        nby = by.sum(1)
        late = np.zeros((n, 10), bool); late[:, s + 1:] = True

        def ref(v):
            b = (v * by).sum(1) / np.maximum(nby, 1)
            lt = (v * late).sum(1) / np.maximum(late.sum(1), 1)
            return np.column_stack([v[ar, p0] - b, v[ar, q0] - b, v[ar, s] - b, lt - b])

        keep = nby >= 1
        for nm, v in (("s", np.full(n, s)), ("g", q["g"][i]), ("Mn", q["Mn"][i]), ("M0", q["M0"][i]), ("S0", q["S0"][i]),
                      ("h4", hs[:, 4]), ("h5", Ha[:, s + 1, 5] if s <= 8 else np.zeros(n)), ("h9", Ha[:, s + 1, 9] if s <= 8 else np.zeros(n))):
            put(nm, v[keep])
        put("act", ref(la)[keep]); put("cf", ref(lc)[keep])
        put("old_cf", ref(sum(oldcf.values()))[keep]); put("N_cf", ref(lc - sum(oldcf.values()))[keep])
        put("old_direct", ref(sum(t[0] for t in told.values()))[keep])
        put("old_via5", ref(sum(t[1] for t in told.values()))[keep])
        put("old_via9", ref(sum(t[2] for t in told.values()))[keep])
        put("p4_direct", ref(t4[0])[keep]); put("p4_via5", ref(t4[1])[keep]); put("p4_via9", ref(t4[2])[keep])
        put("xs", ref(sum(tx))[keep]); put("later", ref(later)[keep])
        for k in PARTS:
            put("conv_" + k, ref(told[k][0] - oldcf[k])[keep])
            put("oldcf_" + k, ref(oldcf[k])[keep])
            put("via59_" + k, ref(told[k][1] + told[k][2])[keep])
        put("ok", (la.argmax(1) == p0)[keep])

R = {k: np.concatenate(v) for k, v in acc.items()}
np.savez(HIDE / "r1_F_record_event" / "s06b_terms.npz", **R)
n = len(R["s"])
COLS = ["p0 (old max)", "q0 (old 2nd)", "s (new max)", "later positions"]
print(f"sequences: n = {n}; network answers p0 in {R['ok'].mean():.4f}")
ROWS = [("cf", "no-record counterfactual, total"),
        ("old_cf", "(1) old state read with no-record gates"),
        ("CONV", "(2a) old state re-read, direct paths: conversion"),
        ("old_via5", "(2b) old state -> unit 5 at s+1 -> ..."),
        ("old_via9", "(2c) old state -> unit 9 at s+1 -> ..."),
        ("p4_direct", "(4a) record pulse h4(s), direct"),
        ("p4_via5", "(4b) record pulse -> unit 5 at s+1 -> ..."),
        ("p4_via9", "(4c) record pulse -> unit 9 at s+1 -> ..."),
        ("xs", "(3a) x_s into wave units"),
        ("later", "(3b) later inputs x_(s+1)..x_9"),
        ("act", "actual logits = sum of (1)..(4)"),
        ("N_cf", "    [inputs x_s'..x_9 in the counterfactual]")]


def get(name):
    return R["old_direct"] - R["old_cf"] if name == "CONV" else R[name]


def table(mask, title):
    print(f"\n=== {title}: n = {mask.sum()} ===   mean logit relative to bystanders")
    print(f"  {'term':50s}" + "".join(f"{c:>18s}" for c in COLS))
    for name, lab in ROWS:
        print(f"  {lab:50s}" + "".join(f"{get(name)[mask, j].mean():18.2f}" for j in range(4)))
    chk = sum(get(nm) for nm in ("old_cf", "CONV", "old_via5", "old_via9", "p4_direct", "p4_via5", "p4_via9", "xs", "later"))
    assert np.allclose(chk, R["act"], atol=1e-6)
    print("  split of (2a) conversion by mode of h_(s-1):")
    for k in PARTS:
        print(f"     {k:47s}" + "".join(f"{R['conv_' + k][mask, j].mean():18.2f}" for j in range(4)))
    print("  split of (2b)+(2c) by mode of h_(s-1):")
    for k in PARTS:
        print(f"     {k:47s}" + "".join(f"{R['via59_' + k][mask, j].mean():18.2f}" for j in range(4)))
    print("  split of (1) by mode of h_(s-1):")
    for k in PARTS:
        print(f"     {k:47s}" + "".join(f"{R['oldcf_' + k][mask, j].mean():18.2f}" for j in range(4)))


g2 = R["g"] >= 2
table(g2, "all s = 3..9, g >= 2")
table(g2 & (R["s"] <= 7), "s = 3..7, g >= 2")
table(R["g"] == 1, "all s = 3..9, g = 1")
for s in (4, 6, 8, 9):
    table(g2 & (R["s"] == s), f"s = {s}, g >= 2")

X = [R["M0"] - R["S0"], R["S0"], R["Mn"] - R["M0"]]
print("\n=== regressions (g >= 2), bystander-referenced: term ~ a (M0-S0) + b S0 + c (Mn-M0)  [no intercept] ===")
for j, c in ((0, "p0"), (1, "q0"), (2, "s ")):
    for name, lab in ROWS[:-1] + [("conv_clock", "    conversion, clock part"), ("conv_level", "    conversion, level part"),
                                  ("conv_runner", "    conversion, runner part")]:
        y = get(name)[g2, j] if name != "conv_clock" and not name.startswith("conv_") else R[name][g2, j]
        b, r2 = ols(y, [v[g2] for v in X])
        print(f"  logit[{c}] {lab:50s}: {b[0]:8.2f} (M0-S0) {b[1]:+8.2f} S0 {b[2]:+8.2f} (Mn-M0)   R2 = {r2:.3f}   mean {y.mean():7.2f}")
    print()

print("=== clock conversion at p0 within (s, g) cells: coefficient of (M0-S0), R2 ===")
print("        " + "".join(f"   g={g}        " for g in range(2, 8)))
for s in range(3, 10):
    row = f"  s={s}: "
    for g in range(2, 8):
        m = (R["s"] == s) & (R["g"] == g)
        if m.sum() < 500:
            row += "      .        "
            continue
        b, r2 = ols(R["conv_clock"][m, 0], [v[m] for v in X])
        row += f" {b[0]:6.1f} ({r2:.2f})  "
    print(row)
print("=== total change at p0 created by the record (act - cf) within (s, g) cells: coefs of (M0-S0), S0, (Mn-M0); R2 ===")
for s in range(3, 10):
    for g in (2, 4, 6):
        m = (R["s"] == s) & (R["g"] == g)
        if m.sum() < 500:
            continue
        y = R["act"][m, 0] - R["cf"][m, 0]
        b, r2 = ols(y, [v[m] for v in X])
        print(f"  s={s} g={g}: {b[0]:6.1f} (M0-S0) {b[1]:+6.1f} S0 {b[2]:+6.1f} (Mn-M0)  R2 = {r2:.2f}   n = {m.sum()}")
