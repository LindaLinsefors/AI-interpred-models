"""D1 / D5. Coefficient vectors per (a1, a2) cell (pooled over t) of  h_t[ORD] ~ g u + S v + T tau + x e,  g = M - S,
shown in mode coordinates / 10.157 as matrices a1 (rows) x a2 (columns).   python -B d1_cells.py [strict|pos] [regs]
Saves the coefficient array to d1_cells_<rule>.npz
"""
from d1_lib import *

rule = sys.argv[1] if len(sys.argv) > 1 else "strict"
regs = (sys.argv[2] if len(sys.argv) > 2 else "g,S,T,x").split(",")
G = load_gram(rule).sum(0)                 # (a1, a2+1, K, K)
NMIN = 2000
B = np.full((10, 10, len(regs), 13), np.nan)
n = np.zeros((10, 10), int)
for a1 in range(10):
    for a2 in range(10):
        n[a1, a2] = G[a1, a2 + 1, 17, 17]
        if n[a1, a2] >= NMIN:
            B[a1, a2] = fit(G[a1, a2 + 1], regs, YH)["B"]
np.savez(HERE / f"d1_cells_{rule}.npz", B=B, n=n, regs=regs)
print(f"rule = {rule}; regressors = {regs}; cells with n >= {NMIN}; rows a1 = 0..9, columns a2 = 0..9")
print("cell counts:\n", n)


def show(k, mode):
    L = LEFT[mode]
    c = B[:, :, k] @ L / 10.157
    print(f"\n--- coefficient of {regs[k]}: mode {mode}" + ("  (|z|@phase deg)" if IS_COMPLEX[mode] else ""))
    for a1 in range(10):
        row = ""
        for a2 in range(10):
            v = c[a1, a2]
            if np.isnan(v):
                row += "      .     " if IS_COMPLEX[mode] else "      .  "
            elif IS_COMPLEX[mode]:
                row += f" {abs(v):5.2f}@{np.degrees(np.angle(v)) % 360:4.0f} "
            else:
                row += f" {v.real:7.3f} "
        print(f"  a1={a1} |{row}")


for k in range(len(regs)):
    if regs[k] == "x":
        continue
    for mode in NAMES[:-2]:
        show(k, mode)
