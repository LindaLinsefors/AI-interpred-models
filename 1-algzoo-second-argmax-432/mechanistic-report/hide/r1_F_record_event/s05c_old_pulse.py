"""s05c. Why is the OLD max's own unit-5 pulse (at p0+1) needed for its later relabelling?
Relabel sequences (record at s, x_s final max, p0 true runner-up), g >= 3, s = 4..9; seed 1095.
Run with unit 5 forced off at step p0+1 only, and compare the state just before the record (s-1) and the final logits."""
from fcommon import *

x = sample(400_000, 1095)
st = running_stats(x)
order = np.argsort(-x, axis=1)
out = []
for s in range(4, 10):
    rec, q = records(x, s, st)
    ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"]) & (q["g"] >= 3) & (q["S0"] > 0)
    i = np.flatnonzero(ok); n = len(i); ar = np.arange(n)
    xi = x[i]; p0, q0 = q["p0"][i], q["q0"][i]
    H, Z = run(xi, return_pre=True)
    h = np.zeros((n, N)); Hm = np.empty((n, 10, N)); Zm = np.empty((n, 10, N))
    for t in range(10):
        z = xi[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        h[:, 5] = np.where(t == p0 + 1, 0.0, h[:, 5])
        Hm[:, t], Zm[:, t] = h, z
    la, lm = H[:, 9] @ U.T, Hm[:, 9] @ U.T
    c0, c1 = coords(H[:, s - 1]), coords(Hm[:, s - 1])
    by = np.ones((n, 10), bool); by[:, s:] = False; by[ar, p0] = False; by[ar, q0] = False
    keep = by.sum(1) >= 1
    ref = lambda v: np.column_stack([v[ar, p0], v[ar, q0], v[ar, s]]) - ((v * by).sum(1) / np.maximum(by.sum(1), 1))[:, None]
    # logits 'just before the record' as the no-record counterfactual would give them
    xc = xi.copy(); xc[:, s] = np.minimum(q["S0"][i], 0) - 1
    lca = logits(run(xc))
    h = np.zeros((n, N))
    for t in range(10):
        z = xc[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        h[:, 5] = np.where(t == p0 + 1, 0.0, h[:, 5])
    lcm = h @ U.T
    d = dict(gated=((Zm[:, s, 4] > 0) & ~(Zm[:, s][:, COMP] > 0).any(1))[keep], ok=(lm.argmax(1) == p0)[keep], ok0=(la.argmax(1) == p0)[keep],
             act=ref(la)[keep], mod=ref(lm)[keep], cfa=ref(lca)[keep], cfm=ref(lcm)[keep],
             h4=(Hm[:, s, 4] / np.maximum(H[:, s, 4], 1e-9))[keep], gap=(q["M0"][i] - q["S0"][i])[keep])
    for nme in NAMES:
        d["rel_" + nme] = (np.abs(c1[nme] - c0[nme]) / (np.abs(c0[nme]) + 1e-9))[keep]
        d["abs_" + nme] = (np.abs(c1[nme] - c0[nme]) / w[4])[keep]
    out.append(d)
R = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
print(f"n = {len(R['ok'])};  P(ans = p0): unmodified {R['ok0'].mean():.3f}, old unit-5 pulse removed {R['ok'].mean():.3f};"
      f"  record at s still gated as a record: {R['gated'].mean():.3f};  h4(s) modified / unmodified: median {np.median(R['h4']):.3f}")
print("change of the mode coordinates at s-1 caused by removing the old unit-5 pulse:  median |dm|/|m|   and median |dm|/10.157")
for nme in NAMES:
    print(f"   {nme:7s}: {np.median(R['rel_' + nme]):7.3f}    {np.median(R['abs_' + nme]):7.3f}")
print("mean logits relative to bystanders          p0       q0        s")
for lab, k in (("no-record cf, unmodified", "cfa"), ("no-record cf, old u5 pulse removed", "cfm"), ("with record, unmodified", "act"), ("with record, old u5 pulse removed", "mod")):
    print(f"   {lab:36s} " + " ".join(f"{v:8.2f}" for v in R[k].mean(0)))
print("created by the record at p0 (with - without record): unmodified %.2f ; old unit-5 pulse removed %.2f" % (
    (R["act"][:, 0] - R["cfa"][:, 0]).mean(), (R["mod"][:, 0] - R["cfm"][:, 0]).mean()))
