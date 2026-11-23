"""D0. First look: write strengths of the exact update rule in tape coordinates, and one value sweep.
Run:  python -B d0_explore.py > d0_explore.txt"""
from dict_common import *

print("Write strengths (complex tape coordinates alt, short, h2, h3; abs@deg).  Gauge: unit 4's pulse writes 1 into each.")
print("  eigenvalues lam          ", fmtc(LAMV, 3))
print("  direct input  l.w[o]     ", fmtc(LEFTM @ w[ORD], 3))
for u in PULSE:
    print(f"  pulse unit {u}  l.W[o,{u}]   ", fmtc(LEFTM @ W[ORD, u], 3))
print("  left vectors on comparators (what a clipped comparator leaves out, per unit of its negative pre-activation):")
for k in COMP:
    print(f"    unit {k}: ", fmtc(LEFTM[:, ORD.index(k)], 3), f"   w[{k}] = {w[k]:.2f}")
print("  w on the pulse units: ", {u: round(w[u], 2) for u in PULSE}, "  W[5,4], W[9,4], W[5,5]:", W[5, 4].round(3), W[9, 4].round(3), W[5, 5].round(3))

# sweep: M = 1 at age 9, S = 0.6 at age 7, test value v at age 4, filler f; final state at step 9
for f in (-1.0, 0.0):
    print(f"\nSweep of the value v of one input at age 4.  M = 1 at age 9, S = 0.6 at age 7, filler {f}.  Contribution = tape(v) - tape(filler).")
    vs = np.round(np.arange(-1.0, 1.0, 0.1), 2)
    X = np.array([seq(10, {9: 1.0, 7: 0.6, 4: v}, f) for v in vs])
    H = run(X); base = run(seq(10, {9: 1.0, 7: 0.6}, f)[None])
    D = zc(H[:, -1]) - zc(base[:, -1])
    print("     v    h5 next step  h4 own step  h9 next |  contribution alt, short, h2, h3 (abs@deg)        | comparators on at own step (1,6,7,8)")
    for i, v in enumerate(vs):
        G = (H[i, 5, COMP] > 0).astype(int)
        print(f"  {v:5.2f}   {H[i, 6, 5]:8.3f}   {H[i, 5, 4]:8.3f}   {H[i, 6, 9]:8.3f} | {fmtc(D[i])} | {G}")
    print("  filler baseline tape (abs@deg):", fmtc(zc(base[:, -1])[0]))
