"""R4 / r05: section 7 interventions on the pulse units: exact detector for unit 4, removal of its
false firings, unit 5 forced off after inputs below the runner-up, both repairs together.
Seeds 7061-7068 (2^21 sequences).  Output: r05_interventions_out.txt
"""
from rlib import *

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


def make_hook(x, Mp, Sp, oracle=False, no_false=False, repair_miss=False, u5_off=False, u5_mode="S", oracle_records_only=False):
    def hook(t, z, h):
        xt = x[:, t]
        M0 = Mp[:, t]
        if oracle:                                   # unit 4 := exact detector at every step
            h[:, 4] = W4 * np.maximum(xt - np.maximum(M0, 0), 0)
        if oracle_records_only:                      # exact value only where x_t is a true positive record
            r = (xt > M0) & (xt > 0)
            h[r, 4] = W4 * (xt[r] - np.maximum(M0[r], 0))
        if no_false:                                 # unit 4 := 0 when x_t is not a new maximum
            h[xt <= M0, 4] = 0.0
        if repair_miss:                              # exact value where x_t is a positive record but unit 4 is silent
            m = (xt > M0) & (xt > 0) & (h[:, 4] <= 0)
            h[m, 4] = W4 * (xt[m] - np.maximum(M0[m], 0))
        if u5_off and t >= 1:                        # unit 5 at t := 0 when x_{t-1} was below the runner-up before it
            S0 = Sp[:, t - 1] if u5_mode == "S" else np.maximum(Sp[:, t - 1], 0)
            lowmask = x[:, t - 1] < S0
            if u5_mode == "S":
                lowmask &= np.isfinite(S0)
            h[lowmask, 5] = 0.0
        return h
    return hook


variants = {
    "(a) unit 4 := exact detector, every step": dict(oracle=True),
    "(a') exact value at true positive records only": dict(oracle_records_only=True),
    "(b1) unit 4 := 0 when x_t <= M_prev": dict(no_false=True),
    "(b2) exact value where a positive record is missed": dict(repair_miss=True),
    "(c) unit 5 := 0 after x_t < S_prev": dict(u5_off=True),
    "(c') same with max(S_prev, 0)": dict(u5_off=True, u5_mode="S+"),
    "(d) = b1 + b2 + c": dict(no_false=True, repair_miss=True, u5_off=True),
    "(b1) + (c)": dict(no_false=True, u5_off=True),
}
tot = {k: np.zeros(5) for k in variants}
cls_tab = {k: np.zeros((4, 3)) for k in ("(b1) unit 4 := 0 when x_t <= M_prev", "(c) unit 5 := 0 after x_t < S_prev", "(d) = b1 + b2 + c")}
base = np.zeros(2)
h4stats = np.zeros(4)
for sd in range(7061, 7069):
    x = gauss(1 << 18, sd)
    st = prefix_stats(x)
    Mp, Sp = prev(st["M"], -np.inf), prev(st["S"], -np.inf)
    pM, pS, pT = st["pM"][:, -1], st["pS"][:, -1], st["pT"][:, -1]
    Mv, Sv, Tv = st["M"][:, -1], st["S"][:, -1], st["T3"][:, -1]
    H0, Z0 = fwd(x)
    a0 = (H0[:, -1] @ U.T).argmax(1)
    ok0 = a0 == pS
    base += [len(x), ok0.sum()]
    # classes for the per-class repair rates
    Sp_, Tp_ = np.maximum(Sv, 0), np.maximum(Tv, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1, g2 = (Mv - Sp_) / Mv, (Sp_ - Tp_) / Mv
    pos = Sv > 0
    c_mr_mfirst = pos & (g1 < 0.05) & (g2 >= 0.05) & (pM < pS)       # max ~ runner-up, max first
    c_mr_rfirst = pos & (g1 < 0.05) & (g2 >= 0.05) & (pS < pM)
    c_rt_rfirst = pos & (g2 < 0.05) & (g1 >= 0.05) & (pS < pT)       # runner-up ~ third, runner-up first
    c_rt_tfirst = pos & (g2 < 0.05) & (g1 >= 0.05) & (pT < pS)
    # unit-4 false firings in the unmodified net
    false4 = (Z0[:, :, 4] > 0) & (x <= Mp)
    h4stats += [false4.any(1).sum(), len(x), false4.sum(), ((x <= Mp) & np.isfinite(Mp)).sum()]
    for name, kw in variants.items():
        Hh, _ = fwd(x, make_hook(x, Mp, Sp, **kw))
        a = (Hh[:, -1] @ U.T).argmax(1)
        ok = a == pS
        tot[name] += [(a == a0).sum(), ok.sum(), (~ok0 & ok).sum(), (ok0 & ~ok).sum(), ((Hh != H0).any((1, 2))).sum()]
        if name in cls_tab:
            for i, c in enumerate((c_mr_mfirst, c_mr_rfirst, c_rt_rfirst, c_rt_tfirst)):
                cls_tab[name][i] += [(c & ~ok0).sum(), (c & ~ok0 & ok).sum(), (c & ok0 & ~ok).sum()]
n, E = base[0], base[0] - base[1]
P("=== SECTION 7 interventions, n = %d (seeds 7061-7068); baseline accuracy %.4f, %d errors" % (n, base[1] / n, E))
P("sequences with a false firing of unit 4: %.4f ; false firings per non-record step: %.4f" % (h4stats[0] / h4stats[1], h4stats[2] / h4stats[3]))
P("%-52s touched  agreement  accuracy  fixed/E  new/E" % "intervention")
for name, t in tot.items():
    P("%-52s %.4f   %.4f     %.4f    %.4f   %.4f" % (name, t[4] / n, t[0] / n, t[1] / n, t[2] / E, t[3] / E))
cn = ["max~runner-up, max first", "max~runner-up, runner-up first", "runner-up~third, runner-up first", "runner-up~third, third first"]
for name, tab in cls_tab.items():
    P("per class under %s: share of all errors / fixed (share of class errors) / new (relative to class errors)" % name)
    for i in range(4):
        P("   %-34s %.4f  %.3f  %.3f" % (cn[i], tab[i, 0] / E, tab[i, 1] / tab[i, 0], tab[i, 2] / tab[i, 0]))
P("-W[4,4] = %.4f  (theta_4(t+1) gets -W[4,4] (x_t - theta_4(t)) from unit 4's own pulse) ; W[4,2]... 0.0970*w4 = %.4f" % (-W[4, 4], -W[4, 4]))

# uniform inputs (report: the two repairs were repeated on Unif(-1,1))
tu = {k: np.zeros(2) for k in ("none", "(b1)", "(c)")}
for sd in (7069, 7070):
    x = np.random.default_rng(sd).uniform(-1, 1, (1 << 19, 10))
    st = prefix_stats(x)
    Mp, Sp = prev(st["M"], -np.inf), prev(st["S"], -np.inf)
    pS = st["pS"][:, -1]
    for k, kw in (("none", {}), ("(b1)", dict(no_false=True)), ("(c)", dict(u5_off=True))):
        a = (fwd(x, make_hook(x, Mp, Sp, **kw))[0][:, -1] @ U.T).argmax(1)
        tu[k] += [len(x), (a == pS).sum()]
P("Unif(-1,1), n = 2^20: accuracy none %.4f ; (b1) %.4f ; (c) %.4f" % tuple(tu[k][1] / tu[k][0] for k in ("none", "(b1)", "(c)")))
out.close()
