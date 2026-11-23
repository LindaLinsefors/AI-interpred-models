"""d8: the decoding result turned round.  If the tape stores V, then a tape rebuilt from V alone must work.
Quiet final states (t = 9).  Fit on the first 75% of the states, test on the rest.
Model of the tape coordinates:  c / M = A_g + B_g . v,   one (A_g, B_g) per class g of ages, v = values in units of M.
    E0  one mean for all states                      E1  g = (a_M, a_S)                 E2  g = (a_M, a_S, a_T)
    E3  g = (a_M, a_S),      v = (S/M)               E4  g = (a_M, a_S, a_T), v = (S/M, T+/M)
    E5  g = (a_M, a_S, a_T), v = (S/M, T+/M, Q+/M)   Q = fourth largest;  T+ = max(T, 0), Q+ = max(Q, 0)
    E6  g = (a_M, a_S, a_T, a_Q), v = (S/M, T+/M, Q+/M)
Classes with fewer than 40 training states fall back to the (a_M, a_S) model of the same kind.
Scores:  share of the variance of c/M explained (all 7 coordinates pooled, each in units of its own variance), and the
FUNCTIONAL test: put the modelled tape M*(A_g + B_g.v) in place of the real one (set_tape) and read the answer.
Output: d8_encoding.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
B = len(x); n_train = int(0.75 * B); tr = np.arange(B) < n_train
M, S, aM, aS, aT, pS = st["M"], st["S"], st["aM"], st["aS"], st["aT"], st["pS"]
srt = np.sort(x, 1)[:, ::-1]; order = np.argsort(-x, 1, kind="stable")
Tp, Qp, aQ = np.maximum(srt[:, 2], 0), np.maximum(srt[:, 3], 0), 9 - order[:, 3]
c = tape_coords(h) / M[:, None]
g2 = aM * 10 + aS; g3 = g2 * 10 + aT; g4 = g3 * 10 + aQ
one = np.ones(B)


def fit(group, V):
    """per-class least squares of c on [1, V]; returns the prediction for every state (classes fitted on train only)."""
    X = np.column_stack([one] + V); pred = np.full(c.shape, np.nan)
    ug, inv = np.unique(group, return_inverse=True)
    o = np.argsort(inv, kind="stable"); cut = np.searchsorted(inv[o], np.arange(len(ug) + 1))
    for k in range(len(ug)):
        idx = o[cut[k]:cut[k + 1]]; t = idx[tr[idx]]
        if len(t) >= 40:
            pred[idx] = X[idx] @ np.linalg.lstsq(X[t], c[t], rcond=None)[0]
    return pred


models = {"E0 one mean": (np.zeros(B, int), []), "E1 (a_M,a_S)": (g2, []), "E2 (a_M,a_S,a_T)": (g3, []),
          "E3 (a_M,a_S) + S/M": (g2, [S / M]), "E4 (a_M,a_S,a_T) + S/M, T+/M": (g3, [S / M, Tp / M]),
          "E5 (a_M,a_S,a_T) + S/M, T+/M, Q+/M": (g3, [S / M, Tp / M, Qp / M]),
          "E6 (a_M,a_S,a_T,a_Q) + S/M, T+/M, Q+/M": (g4, [S / M, Tp / M, Qp / M])}
te = ~tr
var = c[tr].var(0)
out(f"quiet final states {B}; train {n_train}, test {te.sum()}.  Untouched network on the test states: answer = runner-up {(D['ans'] == pS)[te].mean():.3f}")
a0 = answer(set_tape(h, np.zeros_like(c)))[te]
out(f"tape removed (c = 0): answer = runner-up {(a0 == pS[te]).mean():.3f}, same as the untouched answer {(a0 == D['ans'][te]).mean():.3f}")
out("")
out(f"{'model of the tape':40s} {'var. expl.':>10s} | answer = runner-up: {'as is':>6s} {'clipped':>8s} | same answer as untouched {'unit < -0.01 M':>15s} {'fallback':>9s}")
for name, (g, V) in models.items():
    p = fit(g, V); fb = np.isnan(p[:, 0])
    if fb.any(): p[fb] = fit(g2, V[:1] if V else [])[fb]
    ev = 1 - (((c - p)[te] ** 2).mean(0) / var).mean()
    hn = set_tape(h, p * M[:, None])
    a, ac = answer(hn)[te], answer(np.maximum(hn, 0))[te]
    out(f"{name:40s} {ev:10.3f} | {'':19s} {(a == pS[te]).mean():6.3f} {(ac == pS[te]).mean():8.3f} | {(a == D['ans'][te]).mean():24.3f} {(hn.min(1) < -0.01 * M)[te].mean():15.3f} {fb[te].mean():9.3f}")

out("")
out("per coordinate, share of variance explained on the test states (alt, Re short, Im short, Re h2, Im h2, Re h3, Im h3):")
for name in ("E1 (a_M,a_S)", "E4 (a_M,a_S,a_T) + S/M, T+/M", "E6 (a_M,a_S,a_T,a_Q) + S/M, T+/M, Q+/M"):
    g, V = models[name]; p = fit(g, V); fb = np.isnan(p[:, 0])
    if fb.any(): p[fb] = fit(g2, V[:1] if V else [])[fb]
    out(f"   {name:40s}", np.round(1 - ((c - p)[te] ** 2).mean(0) / var, 3))

out("")
out("E4 split by configuration and by the distance of the third: answer = runner-up with the modelled tape / untouched")
g, V = models["E4 (a_M,a_S,a_T) + S/M, T+/M"]; p = fit(g, V); fb = np.isnan(p[:, 0]); p[fb] = fit(g2, V[:1])[fb]
a = answer(set_tape(h, p * M[:, None]))
gap = (S - srt[:, 2]) / M
for nm, k in (("S before M", D["S_before_M"]), ("S after M", ~D["S_before_M"]), ("(S-T)/M < 0.05", gap < .05), ("0.05-0.1", (gap >= .05) & (gap < .1)),
              ("0.1-0.2", (gap >= .1) & (gap < .2)), ("0.2-0.4", (gap >= .2) & (gap < .4)), ("> 0.4", gap >= .4)):
    k = k & te
    out(f"   {nm:16s} n {k.sum():6d}   {(a == pS)[k].mean():.3f} / {(D['ans'] == pS)[k].mean():.3f}")
