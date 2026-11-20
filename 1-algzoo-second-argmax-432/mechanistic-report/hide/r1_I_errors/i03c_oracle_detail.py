"""I3 (detail). Why does the full oracle (a) lower the accuracy?
  (a_rec)  unit 4 := oracle only at true records (x_t > max(M_prev, 0)); misfires untouched
  (a_thr)  unit 4 := 10.157 * (x_t - theta_4(t)) at records (= what the net does) and 0 otherwise  -> identical to (b1')
At a record the net's own pulse is 10.157 (x_t - theta_4) = oracle + 10.157 (M_prev+ - theta_4): it is larger than
the oracle by the sag of the threshold.  New errors of (a_rec) against that sag."""
from icommon import *


def run_arec(x, scale_sag=0.0):
    """unit 4 at true records := oracle + scale_sag * 10.157 * (ref - theta_4) ; scale_sag = 1 reproduces the net at records."""
    B = len(x); h = np.zeros((B, N)); Mprev = np.full(B, -np.inf)
    for t in range(T):
        xt = x[:, t]
        th4 = -(h @ W[4]) / w[4]
        h = np.maximum(xt[:, None] * w + h @ W.T, 0.0)
        ref = np.maximum(Mprev, 0.0)
        rec = xt > ref
        val = W4 * (xt - ref) + scale_sag * W4 * (ref - th4)
        h[:, 4] = np.where(rec, np.maximum(val, 0.0), h[:, 4])
        Mprev = np.maximum(Mprev, xt)
    return h @ U.T


tot = np.zeros(6); n = 0
by_d = {}
for s, x in batches(seeds=SEEDS[:4]):
    tgt = second_argmax(x)
    H = run(x); base = logits(H).argmax(1); ok0 = base == tgt
    lg1 = run_arec(x, 1.0)
    # with scale_sag = 1 the only difference to the net is at records where the net's unit 4 was silent (misses)
    ok_chk = lg1.argmax(1) == tgt
    lg = run_arec(x, 0.0); ok1 = lg.argmax(1) == tgt
    n += len(x)
    tot += [ok0.sum(), ok1.sum(), (~ok0 & ok1).sum(), (ok0 & ~ok1).sum(), (lg1.argmax(1) == base).sum(), ok_chk.sum()]
    # the last record of the sequence: age of the previous max then, and sag of theta_4 then
    rs = running_stats(x)
    pM = x.argmax(1); r = np.arange(len(x))
    prevM = np.where(pM > 0, rs["M1"][r, np.maximum(pM - 1, 0)], -np.inf)
    prevp = np.where(pM > 0, rs["p1"][r, np.maximum(pM - 1, 0)], -1)
    hprev = np.where((pM > 0)[:, None], H[r, np.maximum(pM - 1, 0)], 0.0)
    th4 = -(hprev @ W[4]) / w[4]
    valid = (pM > 0) & (prevM > 0)
    d = pM - prevp
    sag = (prevM - th4) / prevM                        # relative sag of the threshold at the last record
    inc = (x[r, pM] - prevM) / prevM                   # relative increment of the record
    for dd in range(1, 10):
        c = valid & (d == dd)
        v = by_d.setdefault(dd, np.zeros(5))
        v += [c.sum(), ok0[c].sum(), ok1[c].sum(), np.median(sag[c]) * c.sum(), (c & (inc < sag)).sum()]
    c = ~valid
    v = by_d.setdefault("first positive max is the last record", np.zeros(5)); v += [c.sum(), ok0[c].sum(), ok1[c].sum(), 0, 0]

print(f"n = {n}; net accuracy {tot[0]/n:.4f}; (a_rec) accuracy {tot[1]/n:.4f}; errors fixed {tot[2]/(n-tot[0]):.4f}, new errors {tot[3]/(n-tot[0]):.4f} (as shares of the net's errors)")
print(f"check: records := oracle + 10.157*(M_prev+ - theta_4) reproduces the net's answers on {tot[4]/n:.5f} of sequences (differences = repaired misses), accuracy {tot[5]/n:.4f}")
rows = []
for k, v in by_d.items():
    rows.append([k, int(v[0]), f"{v[1]/v[0]:.4f}", f"{v[2]/v[0]:.4f}", f"{v[3]/v[0]:.3f}" if k != "first positive max is the last record" else "-",
                 f"{v[4]/v[0]:.3f}" if k != "first positive max is the last record" else "-"])
print(fmt_table(rows, ["age of the previous max when the final max arrives", "n", "net accuracy", "(a_rec) accuracy", "median relative sag of theta_4", "share with increment < sag"]))
