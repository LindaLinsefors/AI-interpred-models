"""s10. Extra checks for F3/F5/F6 (uses s06c_terms.npz and fresh data, seed 1080).
 1. per-sequence agreement between the created logit at p0 and the clock part of the comparator excess
 2. does the idealised from-weights conversion (s04: resting flow A^j after the clipping) predict the measured
    clock term?  prediction = [U A^(9-s) (E - I) 2Re(lam z_(s-1) R_clock)] at p0 minus bystanders
 3. s = 9: readout with unclipped comparators, restricted to the old positions 0..8
 4. F5: change of the old runner-up's logit by s, in absolute and relative terms
"""
from fcommon import *

R = dict(np.load(HIDE / "r1_F_record_event" / "s06c_terms.npz"))
g2 = R["g"] >= 2
created = R["act"] - R["cf"]
print("=== 1. created logit at p0 vs clock part of the comparator excess (per sequence, g >= 2) ===")
y, c = created[g2, 0], R["C_clock"][g2, 0]
print(f"  corr = {np.corrcoef(y, c)[0, 1]:.3f};  created ~ {np.polyfit(c, y, 1)[0]:.3f} * C_clock + {np.polyfit(c, y, 1)[1]:.2f};"
      f"  mean created {y.mean():.2f}, mean C_clock {c.mean():.2f}, sd of the difference {np.std(y - c):.2f}")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    print(f"    s={s}: corr {np.corrcoef(created[m, 0], R['C_clock'][m, 0])[0, 1]:.3f}  mean created {created[m, 0].mean():6.2f}  mean C_clock {R['C_clock'][m, 0].mean():6.2f}"
          f"  sd(diff) {np.std(created[m, 0] - R['C_clock'][m, 0]):5.2f}")
print("  margin p0 - q0: corr(created, C_clock) =", round(np.corrcoef(created[g2, 0] - created[g2, 1], R["C_clock"][g2, 0] - R["C_clock"][g2, 1])[0, 1], 3))

print("\n=== 2./3./4. fresh data (seed 1080) ===")
x = sample(400_000, 1080)
st = running_stats(x)
H, Z = run(x, return_pre=True)
lg = H[:, 9] @ U.T
order = np.argsort(-x, axis=1)
E = np.eye(13); E[CI, CI] = 0
rows = []
for s in range(3, 10):
    rec, q = records(x, s, st)
    ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"]) & (q["S0"] > 0) & (q["g"] >= 2)
    i = np.flatnonzero(ok); n = len(i); ar = np.arange(n)
    p0, q0 = q["p0"][i], q["q0"][i]
    z = H[i, s - 1][:, ORD] @ LEFT["clock"]
    v = 2 * np.real(np.multiply.outer(LAM["clock"] * z, RIGHT["clock"]))          # propagated clock component, (n,13)
    pred = (v @ (E - np.eye(13)).T) @ np.linalg.matrix_power(A, 9 - s).T @ Uo.T    # (n,10)
    by = np.ones((n, 10), bool); by[:, s:] = False; by[ar, p0] = False; by[ar, q0] = False
    keep = by.sum(1) >= 1
    b = (pred * by).sum(1) / np.maximum(by.sum(1), 1)
    rows.append(dict(s=np.full(n, s)[keep], g=q["g"][i][keep], pred=(pred[ar, p0] - b)[keep], r=np.abs(LAM["clock"] * z)[keep],
                     peak=(np.where(np.arange(10)[None, :] < s, pred, -np.inf).argmax(1) == p0)[keep],
                     gap=(q["M0"][i] - q["S0"][i])[keep]))
P = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
print(f"  2. idealised clock conversion on real clock coordinates (n = {len(P['s'])}): mean predicted bump at p0 (vs bystanders) = {P['pred'].mean():.2f}"
      f"  [measured C_clock mean on the s06c sample: {R['C_clock'][g2, 0].mean():.2f}];  predicted peak among positions < s is at p0 in {P['peak'].mean():.3f}")
print(f"     predicted bump per unit r = |lam z|: mean {np.mean(P['pred'] / P['r']):.2f} (IQR {np.percentile(P['pred'] / P['r'], 25):.2f} .. {np.percentile(P['pred'] / P['r'], 75):.2f});"
      f"  r / (10.157 (M0-S0)): median {np.median(P['r'] / (w[4] * P['gap'])):.2f}")
print("     by s: predicted mean bump | fraction with peak at p0")
for s in range(3, 10):
    m = P["s"] == s
    print(f"       s={s}: {P['pred'][m].mean():6.2f} | {P['peak'][m].mean():.3f}     (measured C_clock mean {R['C_clock'][g2 & (R['s'] == s), 0].mean():6.2f})")

# 3. s = 9, unclipped comparators, old positions only
rec, q = records(x, 9, st)
ok = rec & (order[:, 0] == 9) & (order[:, 1] == q["p0"]) & (q["S0"] > 0)
i = np.flatnonzero(ok); p0, q0 = q["p0"][i], q["q0"][i]
h = H[i, 9].copy(); z9 = Z[i, 9]
h[:, COMP] = np.minimum(z9[:, COMP], h[:, COMP])            # unclipped where clipped
l_un = h @ U.T
print(f"\n  3. record at s = 9 (n = {len(i)}): argmax over the OLD positions 0..8 only")
print(f"     actual readout:      p0 {np.mean(lg[i][:, :9].argmax(1) == p0):.4f}   q0 {np.mean(lg[i][:, :9].argmax(1) == q0):.4f}")
print(f"     comparators unclipped: p0 {np.mean(l_un[:, :9].argmax(1) == p0):.4f}   q0 {np.mean(l_un[:, :9].argmax(1) == q0):.4f}")
ar = np.arange(len(i))
print(f"     mean logit[p0] - logit[q0]: actual {np.mean(lg[i][ar, p0] - lg[i][ar, q0]):.2f}; unclipped {np.mean(l_un[ar, p0] - l_un[ar, q0]):.2f}")

print("\n=== 4. F5: the old runner-up's logit (vs bystanders) without and with the record, by s (g >= 2) ===")
print("    s | no record | with record | change | relative change | sd of change | P(q0 logit lower with the record)")
for s in range(3, 10):
    m = g2 & (R["s"] == s)
    a, c = R["cf"][m, 1], R["act"][m, 1]
    print(f"    {s} | {a.mean():7.2f}   | {c.mean():7.2f}     | {np.mean(c - a):6.2f} | {np.mean(c - a) / a.mean():7.1%}        | {np.std(c - a):6.2f}       | {np.mean(c < a):.3f}")
a, c = R["cf"][g2, 1], R["act"][g2, 1]
print(f"  all | {a.mean():7.2f}   | {c.mean():7.2f}     | {np.mean(c - a):6.2f} | {np.mean(c - a) / a.mean():7.1%}        | {np.std(c - a):6.2f}       | {np.mean(c < a):.3f}")
ap = R["act"][g2, 0]
print(f"  pairwise: P(act[p0] > act[q0]) = {np.mean(ap > c):.4f};  P(act[p0] > cf[q0]) = {np.mean(ap > a):.4f}  (old max's new logit against the runner-up's UNCHANGED logit);"
      f"  P(cf[p0] > act[q0]) = {np.mean(R['cf'][g2, 0] > c):.4f} (old max's old logit against the runner-up's lowered logit)")
