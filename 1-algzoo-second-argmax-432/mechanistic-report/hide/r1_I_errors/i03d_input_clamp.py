"""I3 (supplement, input-level intervention).  Every input that is below the runner-up so far when it arrives
(x_t < S_prev, so it can never be the answer) is replaced by a low value  beta * max(M_prev, 0)  (if that is lower).
The label is unchanged.  This removes every 'spurious candidate' at once and bounds what (b1), (c) can repair.
Also: (f) every input that is not a new max is prevented from firing unit 4 AND, if below S_prev, replaced (clamp + b1)."""
from icommon import *
from i03_interventions import run_iv


def clamp(x, beta):
    B = len(x); y = x.copy()
    Mprev = np.full(B, -np.inf); Sprev = np.full(B, -np.inf)
    for t in range(T):
        xt = x[:, t]
        low = beta * np.maximum(Mprev, 0.0)
        rep = (xt < Sprev) & (Mprev > 0) & (low < xt)
        y[rep, t] = low[rep]
        Sprev = np.where(xt > Mprev, Mprev, np.maximum(Sprev, xt)); Mprev = np.maximum(Mprev, xt)
    return y


res = {}
n = 0; nerr = 0
cls_tot = {}
for s, x in batches(seeds=SEEDS[:4]):
    d = describe(x); tgt = d["pS"]; ok0 = d["ok"]
    n += len(x); nerr += (~ok0).sum()
    M, S, Tv, pM, pS, pT = d["M"], d["S"], d["T"], d["pM"], d["pS"], d["pT"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = (M - np.maximum(S, 0)) / M; g2 = (np.maximum(S, 0) - np.maximum(Tv, 0)) / M
    classes = {"S <= 0": S <= 0, "S~T, S first": (S > 0) & (g2 < 0.05) & (g1 >= 0.05) & (pS < pT), "S~T, T first": (S > 0) & (g2 < 0.05) & (g1 >= 0.05) & (pS > pT),
               "M~S, M first": (S > 0) & (g1 < 0.05) & (g2 >= 0.05) & (pM < pS), "M~S, S first": (S > 0) & (g1 < 0.05) & (g2 >= 0.05) & (pM > pS),
               "both < 0.05": (S > 0) & (g1 < 0.05) & (g2 < 0.05), "min gap >= 0.05": (S > 0) & (np.minimum(g1, g2) >= 0.05)}
    for beta in (0.0, -0.3, -1.0, -3.0):
        y = clamp(x, beta)
        assert np.array_equal(second_argmax(y), tgt)
        for nm, lg in ((f"clamp to {beta} M_prev", logits(run(y))), (f"clamp to {beta} M_prev + (b1)", run_iv(y, {"b1"})[0])):
            a = lg.argmax(1); ok1 = a == tgt
            v = res.setdefault(nm, np.zeros(4)); v += [(a == d["ans"]).sum(), ok1.sum(), (~ok0 & ok1).sum(), (ok0 & ~ok1).sum()]
            for cn, c in classes.items():
                u = cls_tot.setdefault((nm, cn), np.zeros(3)); u += [(c & ~ok0).sum(), (c & ~ok0 & ok1).sum(), (c & ok0 & ~ok1).sum()]

print(f"n = {n}, net accuracy {1 - nerr/n:.4f}")
rows = [[nm, f"{v[0]/n:.4f}", f"{v[1]/n:.4f}", f"{v[2]/nerr:.4f}", f"{v[3]/nerr:.4f}"] for nm, v in res.items()]
print(fmt_table(rows, ["intervention", "agreement with net", "task accuracy", "net's errors fixed", "new errors (share of net's errors)"]))
for nm in ("clamp to -1.0 M_prev", "clamp to -1.0 M_prev + (b1)"):
    print(f"\n{nm}: by class: fixed / new as share of the class's errors")
    print(fmt_table([[cn, int(cls_tot[(nm, cn)][0]), f"{cls_tot[(nm, cn)][1]/cls_tot[(nm, cn)][0]:.3f} / {cls_tot[(nm, cn)][2]/cls_tot[(nm, cn)][0]:.3f}"]
                     for cn in classes], ["class", "errors", "fixed / new"]))
