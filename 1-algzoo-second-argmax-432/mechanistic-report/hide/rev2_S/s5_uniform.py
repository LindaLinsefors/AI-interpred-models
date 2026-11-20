"""Referee S, check 5: the second intervention of the summary (clock pattern removed from the wave units only,
or from the comparators only, one step before a maximum is overtaken) on Unif(-1,1) inputs.
Selection of claim I2 with the same thresholds; N(0,1) shown next to it for comparison."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT

OI = np.array(ORD)
is_comp = np.isin(OI, COMP)
for dist, seed in (("normal", 9501), ("uniform", 9502)):
    x = sample(1_500_000, seed, dist=dist)
    st = running_stats(x); truth = second_argmax(x); ar = np.arange(len(x))
    tr = x.argmax(1); tm = np.maximum(tr - 1, 0)
    p0, q0, M0, S0 = st["p1"][ar, tm], st["p2"][ar, tm], st["M1"][ar, tm], st["M2"][ar, tm]
    ok = (tr >= 3) & (tr <= 8) & (tr - 1 - p0 >= 2) & (S0 > 0.2) & (M0 - S0 > 0.1) & (truth == p0) & (st["p3"][:, 9] == q0)
    x, tr, p0, q0 = x[ok], tr[ok], p0[ok], q0[ok]
    n = len(x)

    def go(part, clip=True):
        h = np.zeros((n, N))
        for t in range(T):
            h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
            sel = np.where(tr - 1 == t)[0]
            if len(sel) and part is not None:
                d = 2 * np.real(np.outer(h[sel][:, OI] @ LEFT["clock"], RIGHT["clock"]))
                if part == "wave":
                    d[:, is_comp] = 0
                elif part == "comp":
                    d[:, ~is_comp] = 0
                hs = h[sel]; hs[:, OI] -= d
                h[sel] = np.maximum(hs, 0) if clip else hs
        a = (h @ U.T).argmax(1)
        return (a == p0).mean(), (a == tr).mean(), (a == q0).mean()

    print(f"\n{dist}: selected {n};  P(old maximum) / P(new maximum) / P(old runner-up), edited state clipped at zero")
    for lab, part in (("unperturbed", None), ("clock pattern removed from all 13 units", "all"),
                      ("removed from the wave units only", "wave"), ("removed from the comparators only", "comp")):
        r = go(part)
        print(f"  {lab:42s} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}")
