"""I4 (closure). A data-free model of the runner-up/third confusions.

For every configuration (p_e, p_l, p_M) and level s = x_e/M in {0.1..0.9}, run the three-value input
    x[p_M] = 1, x[p_e] = s, x[p_l] = s + eps, everything else = b (= -1)
and find the takeover offset eps*(p_e, p_l, p_M, s) where logit[p_l] - logit[p_e] changes sign  [no sampled data].
Then predict, for every N(0,1) census sequence with T > 0, whether the net prefers the third over the runner-up:
    S first (S at p_e, T at p_l):  third wins iff  (T - S)/M > eps*(pS, pT, pM, S/M)
    T first (T at p_e, S at p_l):  third wins iff  (S - T)/M < eps*(pT, pS, pM, T/M)
and compare with what the net does on the real sequence (all ten inputs)."""
from icommon import *
import i04b_takeover as tk

LEV = np.round(np.arange(0.1, 0.91, 0.1), 2)
BG = -1.0
tab = np.full((10, 10, 10, len(LEV)), np.nan)
cfg = [(a, b, c) for a in range(10) for b in range(a + 1, 10) for c in range(10) if c not in (a, b)]
X = []; PE = []; PL = []
for (a, b, c) in cfg:
    for s in LEV:
        x = np.full(10, BG); x[c] = 1.0; x[a] = s
        X.append(x); PE.append(a); PL.append(b)
X = np.array(X); PE = np.array(PE); PL = np.array(PL)
D, _ = tk.sweep(X, PE, PL, "RT", np.ones(len(X)))
eps, nch = tk.crossing(D)
k = 0
for (a, b, c) in cfg:
    for i in range(len(LEV)):
        tab[a, b, c, i] = eps[k]; k += 1
fin = np.isfinite(eps)
print(f"three-value table: {len(cfg)} configurations x {len(LEV)} levels; crossing inside +-0.15: {fin.mean():.4f}; more than one sign change: {np.mean(nch > 1):.4f}")
e5 = tab[..., 4][np.isfinite(tab[..., 4])]
print("eps* at s = 0.5 over the 360 configurations: quantiles 5/25/50/75/95% = " + " / ".join(f"{v:+.4f}" for v in np.quantile(e5, [.05, .25, .5, .75, .95])) +
      f";  mean |eps*| = {np.abs(e5).mean():.4f};  |eps*| > 0.05 in {np.mean(np.abs(e5) > 0.05):.3f} of configurations")
# dependence on the level s: is eps* proportional to (1 - s) = (M - S)/M ?
T9 = tab.reshape(-1, len(LEV)); T9 = T9[np.isfinite(T9).all(1)]
u = 1 - LEV
c_fit = (T9 * u).sum(1) / (u ** 2).sum()
r2p = 1 - ((T9 - c_fit[:, None] * u) ** 2).sum() / ((T9 - T9.mean(1, keepdims=True)) ** 2).sum()
r2tot = 1 - ((T9 - c_fit[:, None] * u) ** 2).sum() / ((T9 - T9.mean()) ** 2).sum()
print(f"[FIT] eps*(config, s) = c(config) * (1 - s): share of the within-configuration variance over s explained: {r2p:.3f}; of the total variance: {r2tot:.3f}")
print("   mean over configurations of eps* at s = 0.1 .. 0.9: " + " ".join(f"{v:+.4f}" for v in T9.mean(0)) + ";  mean |eps*|: " + " ".join(f"{v:.4f}" for v in np.abs(T9).mean(0)))

# ----- predict the census
tot = np.zeros((2, 2, 2))            # [order][predicted third wins][net answers third]
tot_gap = {}
n_all = 0
for sd, x in batches(seeds=SEEDS[:4]):
    d = describe(x)
    pM, pS, pT, M, S, Tv = d["pM"], d["pS"], d["pT"], d["M"], d["S"], d["T"]
    sel = (Tv > 0) & ((M - S) / M >= 0.05)
    n_all += sel.sum()
    pe = np.minimum(pS, pT); pl = np.maximum(pS, pT)
    xe = np.where(pS < pT, S, Tv) / M
    li = np.clip(np.rint((xe - 0.1) / 0.1).astype(int), 0, len(LEV) - 1)
    e = tab[pe, pl, pM, li]
    e = np.where(np.isnan(e), 0.0, e); e = np.clip(e, -0.15, 0.15)
    g2 = (S - Tv) / M
    sfirst = pS < pT
    pred = np.where(sfirst, -g2 > e, g2 < e)
    act = d["ans"] == pT
    for o in (0, 1):
        c = sel & (sfirst == bool(1 - o))
        for p in (0, 1):
            for a in (0, 1):
                tot[o, p, a] += (c & (pred == bool(p)) & (act == bool(a))).sum()
    for lo, hi in ((0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1)):
        c = sel & (g2 >= lo) & (g2 < hi)
        v = tot_gap.setdefault((lo, hi), np.zeros(4)); v += [c.sum(), (c & act).sum(), (c & pred).sum(), (c & act & pred).sum()]
print(f"\nprediction of 'the net answers the third' on N(0,1) sequences with T > 0 and (M - S)/M >= 0.05 (n = {n_all}, seeds 1000..1003)")
for o, nm in enumerate(("S first", "T first")):
    t = tot[o]
    print(f"  {nm}: net answers third {t[:, 1].sum()/t.sum():.4f}; predicted {t[1].sum()/t.sum():.4f}; "
          f"recall (errors predicted) {t[1, 1]/t[:, 1].sum():.3f}; precision {t[1, 1]/t[1].sum():.3f}; agreement {(t[0, 0]+t[1, 1])/t.sum():.4f}")
t = tot.sum(0)
print(f"  both: net answers third {t[:, 1].sum()/t.sum():.4f}; predicted {t[1].sum()/t.sum():.4f}; recall {t[1, 1]/t[:, 1].sum():.3f}; precision {t[1, 1]/t[1].sum():.3f}; agreement {(t[0, 0]+t[1, 1])/t.sum():.4f}")
rows = [[f"[{lo}, {hi})", int(v[0]), f"{v[1]/v[0]:.4f}", f"{v[2]/v[0]:.4f}", f"{v[3]/max(v[1],1):.3f}", f"{v[3]/max(v[2],1):.3f}"] for (lo, hi), v in tot_gap.items()]
print(fmt_table(rows, ["g2 = (S-T)/M", "n", "net answers third", "predicted", "recall", "precision"]))
np.save(HIDE / "r1_I_errors" / "i04e_three_value_table.npy", tab)
