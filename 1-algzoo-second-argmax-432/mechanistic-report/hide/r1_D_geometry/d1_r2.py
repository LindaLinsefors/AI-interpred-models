"""D1 (step 2). R^2 of  h_t[ORD] ~ (M-S) u + S v (+ T, x_t), fitted per a1 and per (a1, a2).  python -B d1_r2.py [strict|pos]"""
from d1_lib import *

rule = sys.argv[1] if len(sys.argv) > 1 else "strict"
G = load_gram(rule)                       # (t, a1, a2+1, K, K)
Ga = G.sum(0)                             # (a1, a2+1, K, K)   pooled over t
print(f"rule = {rule};  states = {int(Ga[..., 17, 17].sum())}")

VARIANTS = [("g,S", ["g", "S"]), ("g,S,T", ["g", "S", "T"]), ("g,S,x", ["g", "S", "x"]), ("g,S,T,x", ["g", "S", "T", "x"])]


def table(title, ys, key):
    print(f"\n### {title}  ({key} R2; rows a1; columns: n, overall, then units {ORD})")
    for split in ("per a1", "per (a1,a2)", "per (t,a1,a2)"):
        for name, regs in VARIANTS:
            print(f"-- fit {split}, regressors {name}")
            for a1 in range(10):
                if split == "per a1":
                    parts = [fit(Ga[a1].sum(0), regs, ys)]
                elif split == "per (a1,a2)":
                    parts = [fit(Ga[a1, k], regs, ys) for k in range(11) if Ga[a1, k, 17, 17] > 50]
                else:
                    parts = [fit(G[t, a1, k], regs, ys) for t in range(10) for k in range(11) if G[t, a1, k, 17, 17] > 50]
                if not parts:
                    continue
                R = r2(parts)
                print(f"  a1={a1} n={int(R['n']):8d}  all={R[key + '_all']:.4f} | " + " ".join(f"{v:6.3f}" for v in R[key]))


table("h_t[ORD]", YH, "c")
table("h_t[ORD] with the echo coordinate removed", YF, "c")
table("h_t[ORD]", YH, "u")
