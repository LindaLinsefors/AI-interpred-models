"""R4 / r09: second sample for the two per-class repair shares of section 7
(64% of 'max ~ runner-up, max first' by removing false firings of unit 4;
 41% of 'runner-up ~ third, runner-up first' by forcing unit 5 off after inputs below the runner-up).
Seeds 7101-7108 (2^21 sequences).  Output: r09_repair_shares_out.txt
"""
from rlib import *

out = open(__file__.replace(".py", "_out.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s); out.write(s + "\n"); out.flush()


tab = np.zeros((2, 2, 3))          # [class, definition of 'first'] x [errors, fixed, new]
acc = np.zeros(4)
for sd in range(7101, 7109):
    x = gauss(1 << 18, sd)
    st = prefix_stats(x)
    Mp, Sp = prev(st["M"], -np.inf), prev(st["S"], -np.inf)
    pM, pS, pT = st["pM"][:, -1], st["pS"][:, -1], st["pT"][:, -1]
    Mv, Sv, Tv = st["M"][:, -1], st["S"][:, -1], st["T3"][:, -1]
    g1 = (Mv - np.maximum(Sv, 0)) / Mv
    g2 = (np.maximum(Sv, 0) - np.maximum(Tv, 0)) / Mv
    pos = Sv > 0

    def h_b1(t, z, h):
        h[x[:, t] <= Mp[:, t], 4] = 0.0
        return h

    def h_c(t, z, h):
        if t >= 1:
            m = (x[:, t - 1] < Sp[:, t - 1]) & np.isfinite(Sp[:, t - 1])
            h[m, 5] = 0.0
        return h

    ok0 = (fwd(x)[0][:, -1] @ U.T).argmax(1) == pS
    okb = (fwd(x, h_b1)[0][:, -1] @ U.T).argmax(1) == pS
    okc = (fwd(x, h_c)[0][:, -1] @ U.T).argmax(1) == pS
    acc += [len(x), ok0.sum(), okb.sum(), okc.sum()]
    for i, (cls, ok) in enumerate(((pos & (g1 < 0.05) & (g2 >= 0.05) & (pM < pS), okb), (pos & (g2 < 0.05) & (g1 >= 0.05) & (pS < pT), okc))):
        tab[i, 0] += [(cls & ~ok0).sum(), (cls & ~ok0 & ok).sum(), (cls & ok0 & ~ok).sum()]
    # the same classes with strict gaps on raw values (third may be negative -> excluded)
    g2r = (Sv - Tv) / Mv
    for i, (cls, ok) in enumerate(((pos & (g1 < 0.05) & (g2r >= 0.05) & (pM < pS), okb), (pos & (Tv > 0) & (g2r < 0.05) & (g1 >= 0.05) & (pS < pT), okc))):
        tab[i, 1] += [(cls & ~ok0).sum(), (cls & ~ok0 & ok).sum(), (cls & ok0 & ~ok).sum()]
E = acc[0] - acc[1]
P("n = %d ; accuracy %.4f ; (b1) %.4f ; (c) %.4f" % (acc[0], acc[1] / acc[0], acc[2] / acc[0], acc[3] / acc[0]))
for i, name in enumerate(("(b1) on 'max~runner-up, max first'", "(c) on 'runner-up~third, runner-up first'")):
    for j, d in enumerate(("gaps with positive parts", "raw third, T>0")):
        e, f, nw = tab[i, j]
        P("  %-42s [%s]: class errors %d (%.4f of all) ; fixed %.4f +- %.4f ; new %.4f" % (name, d, e, e / E, f / e, np.sqrt(f / e * (1 - f / e) / e), nw / e))
out.close()
