"""D1 (step 3). The fitted vectors u(a1), v(a1) of  h_t[ORD] ~ (M-S) u + S v [+ T t + x_t e]  (per a1, pooled over a2 and t),
in unit space and in mode coordinates.   python -B d1_vectors.py [strict|pos] [regs e.g. g,S,T,x]
"""
from d1_lib import *

rule = sys.argv[1] if len(sys.argv) > 1 else "strict"
regs = (sys.argv[2] if len(sys.argv) > 2 else "g,S,T,x").split(",")
G = load_gram(rule).sum(0)                 # (a1, a2+1, K, K)
np.set_printoptions(precision=2, suppress=True, linewidth=250)


def mode_row(vec):
    out = ""
    for n in NAMES:
        c = LEFT[n] @ vec / 10.157
        out += f"{abs(c):7.3f}@{np.degrees(np.angle(c)) % 360:5.0f}" if IS_COMPLEX[n] else f"{c.real:9.3f}    "
    return out


print(f"rule = {rule}, regressors = {regs}; coefficient vectors per unit of value (so an impulse of 1 gives 10.157 into unit 4)")
for ys, yname in ((YH, "h_t[ORD]"), (YZ, "pre-activation z_t[ORD]")):
    for k, rname in enumerate(regs):
        print(f"\n### coefficient of {rname} on {yname}: rows a1, columns units {ORD}")
        for a1 in range(10):
            f = fit(G[a1].sum(0), regs, ys)
            print(f"  a1={a1} | " + " ".join(f"{v:7.2f}" for v in f["B"][k]))
for k, rname in enumerate(regs):
    print(f"\n### coefficient of {rname} on h_t[ORD], in mode coordinates / 10.157: rows a1; columns " + "  ".join(NAMES))
    for a1 in range(10):
        f = fit(G[a1].sum(0), regs, YH)
        print(f"  a1={a1} | " + mode_row(f["B"][k]))
