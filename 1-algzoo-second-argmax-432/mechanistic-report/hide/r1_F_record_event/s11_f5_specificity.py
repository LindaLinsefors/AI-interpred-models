"""s11 (task F5). Is the change of the old runner-up's logit at a record specific to q0?
Compare the record-induced change (actual - no-record counterfactual) at q0 with the change at one bystander
(a position < s other than p0, q0), both measured against the mean change of the remaining bystanders.
Same selection as s06 (record at s, x_s final max, p0 final runner-up, S0 > 0, g >= 2); seed 1090; needs >= 3 bystanders."""
from fcommon import *

x = sample(400_000, 1090)
st = running_stats(x)
order = np.argsort(-x, axis=1)
rng = np.random.default_rng(1090)
out = []
for s in range(5, 10):
    rec, q = records(x, s, st)
    ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"]) & (q["S0"] > 0) & (q["g"] >= 2)
    i = np.flatnonzero(ok); n = len(i); ar = np.arange(n)
    xa = x[i]; xc = xa.copy(); xc[:, s] = np.minimum(q["S0"][i], 0) - 1
    la, lc = logits(run(xa)), logits(run(xc))
    d = la - lc
    p0, q0 = q["p0"][i], q["q0"][i]
    by = np.ones((n, 10), bool); by[:, s:] = False; by[ar, p0] = False; by[ar, q0] = False
    # pick one random bystander b; reference = the other bystanders
    r = rng.random((n, 10)); r[~by] = -1
    b = r.argmax(1)
    ref = by.copy(); ref[ar, b] = False
    refmean = lambda v: (v * ref).sum(1) / ref.sum(1)
    # is the bystander the third-largest-so-far (rank 3) or lower?
    xb = xa[ar, b]
    out.append(dict(s=np.full(n, s), dq=d[ar, q0] - refmean(d), db=d[ar, b] - refmean(d), dp=d[ar, p0] - refmean(d),
                    lq=lc[ar, q0] - refmean(lc), lb=lc[ar, b] - refmean(lc), xb=xb, S0=q["S0"][i]))
R = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
print(f"n = {len(R['s'])} (s = 5..9, g >= 2)")
print("change created by the record, relative to the mean change of the other bystanders:")
for lab, k in (("old max p0", "dp"), ("old runner-up q0", "dq"), ("one random bystander", "db")):
    v = R[k]
    print(f"   {lab:22s}: mean {v.mean():7.2f}   sd {v.std():6.2f}   median {np.median(v):7.2f}   P(<0) {np.mean(v < 0):.3f}")
print(f"no-record logit vs the other bystanders: q0 mean {R['lq'].mean():.2f} (sd {R['lq'].std():.2f});  random bystander mean {R['lb'].mean():.2f} (sd {R['lb'].std():.2f})")
b, r2 = ols(R["dq"], [R["lq"]], intercept=True)
print(f"change at q0 ~ {b[0]:.3f} * (its no-record logit) + {b[1]:.2f}   R2 = {r2:.3f}   (a slope of -1 would mean full erasure)")
b, r2 = ols(R["db"], [R["lb"]], intercept=True)
print(f"change at a bystander ~ {b[0]:.3f} * (its no-record logit) + {b[1]:.2f}   R2 = {r2:.3f}")
for s in range(5, 10):
    m = R["s"] == s
    bq = ols(R["dq"][m], [R["lq"][m]], intercept=True)[0]
    bb = ols(R["db"][m], [R["lb"][m]], intercept=True)[0]
    print(f"   s={s}: slope at q0 {bq[0]:6.3f} (intercept {bq[1]:6.2f}); slope at a bystander {bb[0]:6.3f} (intercept {bb[1]:6.2f});  mean change q0 {R['dq'][m].mean():6.2f}, bystander {R['db'][m].mean():6.2f}")
