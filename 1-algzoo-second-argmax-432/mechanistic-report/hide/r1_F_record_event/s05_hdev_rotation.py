"""s05 (task F3 ii). Counterfactuals on data: change mode coordinates of h_{s-1} (one step before a record at s),
then let the network run on the true inputs x_s..x_9.

Sequences: record at s (2..9), M0 > 0, x_s is the final max and the old max (position p0, age g at step s)
is the true final runner-up.  N(0,1), seeds 1030, 1031 (400k each).

Interventions on h_{s-1}[ORD] (mode coordinates from modes.py):
   zero:    z_n -> 0
   rot k:   z_n -> z_n * exp(i k angle(lam_n))   (k 'resting steps' forward; for alt: (-1)^k)
for mode sets: clock | h2 | h3 | short | alt | all rotating (clock,h2,h3,short,alt) | all rotating except clock.
If the clock holds the old max's age and the record turns it into the answer, 'rot k' of the clock should move
the final answer from p0 to p0 - k.
"""
from fcommon import *

SETS = {"clock": ["clock"], "h2": ["h2"], "h3": ["h3"], "short": ["short"], "alt": ["alt"],
        "all5": ["clock", "h2", "h3", "short", "alt"], "all-but-clock": ["h2", "h3", "short", "alt"]}
KS = (-2, -1, 1, 2)


def intervene(h, names, k):
    """h (B,16) -> modified copy. k = None: zero the coordinates; else rotate by k steps."""
    h = h.copy()
    ho = h[:, ORD]
    d = 0
    for n in names:
        z = ho @ LEFT[n]
        if k is None:
            z2 = 0 * z
        elif IS_COMPLEX[n]:
            z2 = z * np.exp(1j * k * np.angle(LAM[n]))
        else:
            z2 = z * (np.sign(LAM[n].real) ** k)
        dz = z2 - z
        d = d + (2 * np.real(np.multiply.outer(dz, RIGHT[n])) if IS_COMPLEX[n] else np.real(np.multiply.outer(dz, RIGHT[n])))
    h[:, ORD] = ho + d
    return h


rows = {}
def add(key, **kw):
    r = rows.setdefault(key, {})
    for a, b in kw.items():
        r.setdefault(a, []).append(b)

for seed in (1030, 1031):
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    lg = logits(H)
    order = np.argsort(-x, axis=1)
    for s in range(2, 10):
        rec, q = records(x, s, st)
        ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"])
        i = np.flatnonzero(ok)
        p0, q0, g = q["p0"][i], q["q0"][i], q["g"][i]
        n = len(i)
        ar = np.arange(n)
        base = lg[i]
        basec = base - base.mean(1, keepdims=True)
        hprev = H[i, s - 1]
        for sname, names in SETS.items():
            for k in (None,) + KS:
                h2 = intervene(hprev, names, k)
                Hc, Zc = run_from(h2, x[i], s)
                l2 = Hc[:, -1] @ U.T
                l2c = l2 - l2.mean(1, keepdims=True)
                ans = l2.argmax(1)
                tgt = p0 - (k or 0)
                valid = (tgt >= 0) & (tgt <= s - 1) & (tgt != q0) if k else np.ones(n, bool)
                tg = np.clip(tgt, 0, 9)
                add((sname, k), s=np.full(n, s), g=g, valid=valid, base_ok=base.argmax(1) == p0,
                    a_p0=ans == p0, a_tgt=ans == tg, a_s=ans == s, a_q0=ans == q0,
                    base_tgt=base.argmax(1) == tg,
                    dl_p0=l2c[ar, p0] - basec[ar, p0], dl_tgt=l2c[ar, tg] - basec[ar, tg],
                    l_p0=l2c[ar, p0], l_tgt=l2c[ar, tg], b_p0=basec[ar, p0],
                    neg=(h2 < -1e-9).any(1), recgate=(Zc[:, 0, 4] > 0) & ~(Zc[:, 0][:, COMP] > 0).any(1),
                    gap=q["M0"][i] - q["S0p"][i], M0=q["M0"][i])

R = {key: {a: np.concatenate(b) for a, b in r.items()} for key, r in rows.items()}
r0 = R[("clock", None)]
print(f"sequences: n = {len(r0['s'])};  network (no intervention) answers p0 in {r0['base_ok'].mean():.4f};  by g: " +
      ", ".join(f"g={g}: {r0['base_ok'][r0['g'] == g].mean():.3f} (n={int((r0['g'] == g).sum())})" for g in range(1, 8)))

print("\n=== A. zeroing a mode at step s-1: where does the final answer go?  (g >= 2) ===")
print("  mode set        P(ans=p0)  P(ans=s)  P(ans=q0)  other | mean change of centred logit at p0 | mean centred logit at p0: before -> after | record still gated as record | state has negative entries")
for sname in SETS:
    r = R[(sname, None)]
    m = r["g"] >= 2
    oth = 1 - r["a_p0"][m].mean() - r["a_s"][m].mean() - r["a_q0"][m].mean()
    print(f"  {sname:14s}  {r['a_p0'][m].mean():8.3f}  {r['a_s'][m].mean():8.3f}  {r['a_q0'][m].mean():8.3f}  {oth:6.3f} | {r['dl_p0'][m].mean():8.2f} |"
          f" {r['b_p0'][m].mean():6.2f} -> {r['l_p0'][m].mean():6.2f} | {r['recgate'][m].mean():.3f} | {r['neg'][m].mean():.3f}")
print("  by g for the clock:")
for g in range(1, 8):
    r = R[("clock", None)]
    m = r["g"] == g
    print(f"     g={g}: P(ans=p0) {r['a_p0'][m].mean():.3f}  P(ans=s) {r['a_s'][m].mean():.3f}  P(ans=q0) {r['a_q0'][m].mean():.3f}"
          f"   d logit[p0] {r['dl_p0'][m].mean():7.2f}   (baseline P(ans=p0) {r['base_ok'][m].mean():.3f})")

print("\n=== B. rotating by k steps at step s-1: does the answer move to p0 - k?  (g >= 2; only sequences where p0-k is a valid earlier position != q0) ===")
print("  mode set        k     n      P(ans=p0-k)  P(ans=p0)  P(ans=s)  P(ans=q0) | d logit[p0]  d logit[p0-k] | gated as record | neg entries")
for sname in SETS:
    for k in KS:
        r = R[(sname, k)]
        m = (r["g"] >= 2) & r["valid"]
        print(f"  {sname:14s} {k:+d}  {m.sum():6d}   {r['a_tgt'][m].mean():8.3f}    {r['a_p0'][m].mean():8.3f}  {r['a_s'][m].mean():8.3f}  {r['a_q0'][m].mean():8.3f} |"
              f" {r['dl_p0'][m].mean():8.2f}   {r['dl_tgt'][m].mean():8.2f}   | {r['recgate'][m].mean():.3f} | {r['neg'][m].mean():.3f}")
print("  (baseline without intervention: P(ans = p0-k) = " +
      ", ".join(f"k={k:+d}: {R[('clock', k)]['base_tgt'][(R[('clock', k)]['g'] >= 2) & R[('clock', k)]['valid']].mean():.4f}" for k in KS) + ")")

print("\n=== C. clock rotation, split by g and by s (k = +1 and -1): P(ans = p0-k) ===")
for k in (1, -1):
    r = R[("clock", k)]
    print(f"  k={k:+d} by g: " + "  ".join(f"g={g}: {r['a_tgt'][(r['g'] == g) & r['valid']].mean():.3f} (n={int(((r['g'] == g) & r['valid']).sum())})" for g in range(1, 8)))
    print(f"  k={k:+d} by s (g>=2): " + "  ".join(f"s={s}: {r['a_tgt'][(r['g'] >= 2) & (r['s'] == s) & r['valid']].mean():.3f}" for s in range(3, 10)))
    rr = R[("all5", k)]
    print(f"  all5 k={k:+d} by s (g>=2): " + "  ".join(f"s={s}: {rr['a_tgt'][(rr['g'] >= 2) & (rr['s'] == s) & rr['valid']].mean():.3f}" for s in range(3, 10)))

print("\n=== D. clock rotation by relative gap (M0-S0+)/M0 terciles (g >= 2, k = +1): P(ans = p0-1) ===")
r = R[("clock", 1)]
m = (r["g"] >= 2) & r["valid"]
rel = r["gap"] / r["M0"]
qs = np.quantile(rel[m], [1 / 3, 2 / 3])
for lab, mm in (("small gap", rel < qs[0]), ("middle", (rel >= qs[0]) & (rel < qs[1])), ("large gap", rel >= qs[1])):
    print(f"   {lab:10s}: P(ans=p0-1) {r['a_tgt'][m & mm].mean():.3f}   P(ans=p0) {r['a_p0'][m & mm].mean():.3f}   P(ans=q0) {r['a_q0'][m & mm].mean():.3f}   n={int((m & mm).sum())}")
