"""s05b (task F3 ii, controls; and 'unit 5's earlier pulse').  Relabel sequences as in s05 (record at s, M0 > 0,
x_s final max, old max p0 = true final runner-up), g >= 2.  Seeds 1030, 1031 (same as s05).

 A. direction control for the rotations: P(ans = p0 - k) against P(ans = p0 + k) (the wrong direction)
 B. zeroing a mode at s-1, conditioned on the record still being gated as a record
    (unit 4 on and all four comparators off at step s)
 C. whose pulses matter: unit 5 / unit 9 forced off, unit 0 forced linear, separately after the OLD max's own
    arrival (steps p0+1, p0+2) and after the NEW record (steps s+1, s+2)
"""
from fcommon import *
from s05_hdev_rotation import intervene, SETS      # noqa  (re-runs s05's prints; ignore that output)

print("\n\n################ s05b ################")
acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)


def run_mask(x, off=None, lin=None):
    B, Tn = x.shape
    h = np.zeros((B, N))
    for t in range(Tn):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        for u, m in (lin or {}).items():
            h[:, u] = np.where(m[:, t], z[:, u], h[:, u])
        for u, m in (off or {}).items():
            h[:, u] = np.where(m[:, t], 0.0, h[:, u])
    return h @ U.T


for seed in (1030, 1031):
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    lg = logits(H)
    order = np.argsort(-x, axis=1)
    for s in range(3, 10):
        rec, q = records(x, s, st)
        ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"]) & (q["g"] >= 2)
        i = np.flatnonzero(ok); n = len(i); ar = np.arange(n)
        p0, q0, g = q["p0"][i], q["q0"][i], q["g"][i]
        put("s", np.full(n, s)); put("g", g); put("base", lg[i].argmax(1) == p0)
        hprev = H[i, s - 1]
        for sname in ("clock", "all5", "all-but-clock", "h2", "h3", "short", "alt"):
            for k in ((None, 1, -1) if sname in ("clock", "all5") else (None,)):
                Hc, Zc = run_from(intervene(hprev, SETS[sname], k), x[i], s)
                a = (Hc[:, -1] @ U.T).argmax(1)
                gated = (Zc[:, 0, 4] > 0) & ~(Zc[:, 0][:, COMP] > 0).any(1)
                if k is None:
                    put(f"z_{sname}_ok", a == p0); put(f"z_{sname}_gated", gated)
                else:
                    v = (p0 - k >= 0) & (p0 - k <= s - 1) & (p0 + k >= 0) & (p0 + k <= s - 1) & (p0 - k != q0) & (p0 + k != q0)
                    put(f"r_{sname}_{k}_valid", v); put(f"r_{sname}_{k}_with", a == p0 - k); put(f"r_{sname}_{k}_against", a == p0 + k)
        # C. pulse-specific interventions
        T = np.arange(10)[None, :]
        xi = x[i]
        for lab, t0 in (("old", p0), ("new", np.full(n, s))):
            m1 = T == (t0[:, None] + 1)
            m2 = T == (t0[:, None] + 2)
            if lab == "old":                     # only steps strictly before the new record
                m1 &= T < s; m2 &= T < s
            for nm, kw in ((f"u5off_{lab}", dict(off={5: m1})), (f"u9off_{lab}", dict(off={9: m1})),
                           (f"u59off_{lab}", dict(off={5: m1, 9: m1})), (f"u0lin_{lab}", dict(lin={0: m2}))):
                a = run_mask(xi, **kw).argmax(1)
                put(nm + "_ok", a == p0); put(nm + "_max", a == s); put(nm + "_q0", a == q0); put(nm + "_nb", np.abs(a - p0) == 1)
        put("has_old2", (p0 + 2 < s)); put("old5_on", H[i, np.minimum(p0 + 1, 9), 5] > 0); put("old0_off", Z[i, np.minimum(p0 + 2, 9), 0] <= 0)

R = {k: np.concatenate(v) for k, v in acc.items()}
n = len(R["s"])
print(f"n = {n} relabel sequences with g >= 2; unmodified P(ans = p0) = {R['base'].mean():.4f}")
print("\n=== A. rotation direction control (only sequences where both p0-k and p0+k are valid earlier positions != q0) ===")
for sname in ("clock", "all5"):
    for k in (1, -1):
        v = R[f"r_{sname}_{k}_valid"]
        print(f"   {sname:6s} k={k:+d}: n = {v.sum():6d}   P(ans = p0-k) [with the rotation] = {R[f'r_{sname}_{k}_with'][v].mean():.3f}"
              f"   P(ans = p0+k) [against] = {R[f'r_{sname}_{k}_against'][v].mean():.3f}")

print("\n=== B. zeroing a mode at s-1: P(ans = p0), all / only where the record is still gated as a record ===")
for sname in ("clock", "all5", "all-but-clock", "h2", "h3", "short", "alt"):
    gt = R[f"z_{sname}_gated"]
    print(f"   {sname:14s}: all {R[f'z_{sname}_ok'].mean():.3f}   gated ({gt.mean():.3f} of cases) {R[f'z_{sname}_ok'][gt].mean():.3f}   not gated {R[f'z_{sname}_ok'][~gt].mean():.3f}")

print("\n=== C. pulses of the OLD max (steps p0+1, p0+2 < s) vs pulses of the NEW record (s+1, s+2): P(ans = p0) and where it goes ===")
print(f"   (unit 5 fired at p0+1 in {R['old5_on'].mean():.3f}; unit 0 was off at p0+2 in {R['old0_off'][R['has_old2']].mean():.3f} of the cases with p0+2 < s)")
for lab in ("old", "new"):
    for nm, txt in (("u5off", "unit 5 off at +1"), ("u9off", "unit 9 off at +1"), ("u59off", "units 5, 9 off at +1"), ("u0lin", "unit 0 linear at +2")):
        k = f"{nm}_{lab}"
        m = R["has_old2"] if (lab == "old" and nm == "u0lin") else np.ones(n, bool)
        if lab == "new":
            m = R["s"] <= (8 if nm != "u0lin" else 7)
        print(f"   {lab} max, {txt:22s}: n = {m.sum():6d}  P(ans = p0) {R[k + '_ok'][m].mean():.3f}  [unmodified {R['base'][m].mean():.3f}]"
              f"   ans = s {R[k + '_max'][m].mean():.3f}   ans = q0 {R[k + '_q0'][m].mean():.3f}   ans = p0+-1 {R[k + '_nb'][m].mean():.3f}")
