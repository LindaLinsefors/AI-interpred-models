"""I5 (detail). Is the loss for an old max due to the sagging threshold of unit 4?  Accuracy by a1 (age of the max) and by
a2 (age of the runner-up) for the net and under (b1) 'unit 4 off unless new max' and (c) 'unit 5 off after x_t < S_prev'.
Also the error rate by a1 for sequences in which unit 4 misfired at least once vs never."""
from icommon import *
from i03_interventions import run_iv

acc = {k: np.zeros((2, 10)) for k in ("net", "b1", "c", "d")}
cnt = np.zeros((2, 10)); mis = np.zeros((2, 10, 2))            # [with misfire?][a1][n, errors]
mm = np.zeros((10, 3))                                           # by a1: errors answering the max: n, fixed by b1, misfire on the runner-up
for s, x in batches(seeds=SEEDS[:4]):
    d = describe(x)
    pM, pS, ok = d["pM"], d["pS"], d["ok"]
    a1, a2 = 9 - pM, 9 - pS
    H = run(x)
    Mprev = np.concatenate([np.full((len(x), 1), -np.inf), np.maximum.accumulate(x, 1)[:, :-1]], 1)
    misfire = (H[..., 4] > 0) & (x <= np.maximum(Mprev, 0))
    anym = misfire.any(1)
    oks = {"net": ok}
    for k, mode in (("b1", {"b1"}), ("c", {"c"}), ("d", {"b1", "b2", "c"})):
        oks[k] = run_iv(x, mode)[0].argmax(1) == pS
    for a in range(10):
        cnt[0, a] += (a1 == a).sum(); cnt[1, a] += (a2 == a).sum()
        for k in acc:
            acc[k][0, a] += oks[k][a1 == a].sum(); acc[k][1, a] += oks[k][a2 == a].sum()
        for m in (0, 1):
            c = (a1 == a) & (anym == bool(m))
            mis[m, a] += [c.sum(), (~ok[c]).sum()]
        e = (a1 == a) & ~ok & (d["rank"] == 1)
        mm[a] += [e.sum(), (e & oks["b1"]).sum(), (e & misfire[np.arange(len(x)), pS]).sum()]

for i, nm in enumerate(("a1 = age of the max", "a2 = age of the runner-up")):
    print(f"\n## accuracy by {nm}")
    rows = [[a] + [f"{acc[k][i, a]/cnt[i, a]:.4f}" for k in ("net", "b1", "c", "d")] + [f"{(acc['d'][i, a]-acc['net'][i, a])/cnt[i, a]*100:+.2f}"] for a in range(10)]
    print(fmt_table(rows, ["age", "net", "(b1)", "(c)", "(d)", "gain of (d), pp"]))
print("\n## by a1: share of sequences with at least one misfire of unit 4, and error rates with / without")
rows = [[a, f"{mis[1, a, 0]/(mis[0, a, 0]+mis[1, a, 0]):.4f}", f"{mis[1, a, 1]/mis[1, a, 0]:.4f}", f"{mis[0, a, 1]/mis[0, a, 0]:.4f}",
         f"{mis[1, a, 1]/(mis[0, a, 1]+mis[1, a, 1]):.3f}"] for a in range(10)]
print(fmt_table(rows, ["a1", "share with a misfire", "error rate with misfire", "error rate without", "share of this row's errors that had a misfire"]))
print("\n## by a1: errors that answer the max: count, share repaired by (b1), share where unit 4 misfired on the runner-up itself")
rows = [[a, int(mm[a, 0]), f"{mm[a, 1]/mm[a, 0]:.3f}", f"{mm[a, 2]/mm[a, 0]:.3f}"] for a in range(10)]
print(fmt_table(rows, ["a1", "errors answering the max", "repaired by (b1)", "unit 4 fired on the runner-up"]))
