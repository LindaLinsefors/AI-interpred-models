"""Checks of the weight-derived numbers in the decomposition-route working paper
("comparators, a recorder and a rotating tape") against the checkpoint. The paper's value is in brackets.

Read-only on the weights. Run:  python -B check_decomposition_paper_weights.py   (instant)
"""
from pathlib import Path
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=220)
ROOT = Path(__file__).resolve().parents[4]                # workspace root
Z = np.load(ROOT / "reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10_weights.npz")
w, Whh, Wout = Z["W_ih"][:, 0].astype(float), Z["W_hh"].astype(float), Z["W_out"].astype(float)
CMP = [4, 7, 6, 1, 8]
mpow = np.linalg.matrix_power


def D(off):
    """Diagonal 0/1 gate pattern with the listed units off."""
    d = np.ones(16); d[list(off)] = 0
    return np.diag(d)


e4 = np.eye(16)[4]
A = D([4, 5, 9]) @ Whh                                     # map of the resting pattern (n4, n5, n9 off)

print("Table 1 input weights w_j:", " ".join(f"n{j}:{w[j]:+.2f}" for j in range(16)))
print(f"\nWhh[2,2] {Whh[2, 2]:.3f} [0.97], Whh[2,4] {Whh[2, 4]:.3f} [1.73], Whh[0,4] {Whh[0, 4]:.3f} [0.59]")
print(f"Whh[5,4] {Whh[5, 4]:.3f} [1.19] -> x w4 = {Whh[5, 4] * w[4]:.2f} [12.1]")
print(f"Whh[5,7] {Whh[5, 7]:.3f} [-0.59] -> x w7 = {Whh[5, 7] * w[7]:.2f} [7.8];  Whh[5,8] {Whh[5, 8]:.3f} [-0.85] -> x w8 = {Whh[5, 8] * w[8]:.2f} [10.5];"
      f"  through n6 and n14: {Whh[5, 6] * w[6] + Whh[5, 14] * w[14]:.2f} [-0.4]")
print(f"Wout[9,4] {Wout[9, 4]:.3f} [-4.99] -> x w4 = {Wout[9, 4] * w[4]:.1f} [-50.7];  Wout[9,7] {Wout[9, 7]:.3f} [-3.79] -> x 13.17 = {Wout[9, 7] * -w[7]:.1f}")
print(f"not quoted in the paper: Whh[9,7] {Whh[9, 7]:.2f}, Whh[9,8] {Whh[9, 8]:.2f} (hold n9 off); Whh[0,7] {Whh[0, 7]:.2f}, Whh[0,9] {Whh[0, 9]:.2f} (switch n0 off)")
for x0 in (1.0, -1.0):
    print(f"step-1 threshold of n4 for x0 = {x0:+.0f}: {-Whh[4] @ np.maximum(w * x0, 0) / w[4]:+.4f} [0.99 x0 if x0 > 0, -0.002 x0 if x0 < 0]")

print("\nRouting to the recorder, Whh D_t w (per unit of the input, units 0..15):")
print("  n1 n6 n7 n8 all on:", Whh @ D([4, 5, 9]) @ w, "[18.0 on n5, 84.3 on n9, at most 0.4 elsewhere]")
print("  n8 off            :", Whh @ D([4, 5, 9, 8]) @ w, "[7.5 on n5, 70.9 on n9; n3 n0 n15 n8: -13.5 -10.2 -9.1 +7.7]")
for off in ([1], [6], [1, 6], [1, 8], [6, 8], [1, 6, 8]):
    v = Whh @ D([4, 5, 9] + off) @ w
    print(f"  n{off} off: n5 {v[5]:.1f}, largest elsewhere (not n9) {np.abs(np.delete(v, [5, 9])).max():.1f}   [7.5-18.2 on n5, up to 19.5 elsewhere]")

lam, V = np.linalg.eig(A)
keep = np.abs(lam) > 0.5
print("\neigenvalues of the resting map A with modulus > 0.5 (modulus, angle in degrees):")
print("  ", sorted({(round(float(abs(l)), 3), round(float(abs(np.degrees(np.angle(l)))), 2)) for l in lam[keep]}, reverse=True))
print("   [1.01 and 0.97 real, 0.95 at 43, 0.94 at 83, -0.93 real, 0.84 at 125, 0.74 at 43]")

r = D([4, 9]) @ Whh @ D([4, 5, 9]) @ w                     # record of an input that passed no threshold
print("\nFig 4a, final logits Wout A^(8-p) r per unit of a recorded input at position p = 2..8 (columns: logit positions 0..9):")
for p in range(2, 9):
    print(f"  p={p}:", Wout @ mpow(A, 8 - p) @ r)
print("  [own effect 72 76 71 70 67 62 62; about +20 one position later]")

print("\ncarriers of the record's own effect, Wout[9-a, n] (A^(a-1) r)_n, four largest per age a:")
for a in range(1, 8):
    tr = mpow(A, a - 1) @ r
    c = Wout[9 - a] * tr
    print(f"  age {a}: own {c.sum():+.1f}; " + ", ".join(f"n{n}: {c[n]:+.1f} = {Wout[9 - a, n]:+.2f} x {tr[n]:+.1f}" for n in np.argsort(-np.abs(c))[:4]))
print("  [n0 at age 2; n3, n12, n13 at age 3; n12 at age 4; n11 at age 5; n10 and n15 at age 6]")

# split a trace along A's eigenvectors, grouped into real modes and conjugate pairs
names = ["null" if abs(l) < 1e-6 else f"{l.real:+.2f}" if abs(l.imag) < 1e-9 else f"{abs(l):.2f}@{abs(np.degrees(np.angle(l))):.0f}" for l in lam]
groups = sorted(set(names), key=lambda k: -max(abs(lam[i]) for i in range(16) if names[i] == k))
idx = {k: [i for i in range(16) if names[i] == k] for k in groups}
cr = np.linalg.solve(V, r.astype(complex))
print("\neigen-split of the record's own effect by age (columns: " + ", ".join(groups) + "):")
for a in range(1, 8):
    print(f"  age {a}:", np.array([sum(cr[i] * lam[i] ** (a - 1) * (Wout[9 - a] @ V[:, i]) for i in idx[k]).real for k in groups]))
print("  [ages 2-6: mostly the pairs 0.94@83, 0.84@125, 0.74@43, partly -0.93; the pair 0.95@43 below 1 per unit; fast modes at age 1]")

v = w[4] * D([4]) @ Whh @ e4                               # maximum's trace one step after its arrival
print(f"\n|w4 D Whh e4| = {np.linalg.norm(v):.2f} [40.7]")
print("derived gains on the maximum, -Whh[j] . w4 A^(a-2) D Whh e4 / w_j, rows n4 n7 n6 n1 n8, ages 2..9:")
for j in CMP:
    print(f"  n{j}:", np.array([-Whh[j] @ mpow(A, a - 2) @ v / w[j] for a in range(2, 10)]))
print("  [n4 0.98 at age 2 and 0.72 at age 7; n6, n1, n8 peak at ages 4, 6 and 7-8]")

cv = np.linalg.solve(V, v.astype(complex))
print("\nNOT IN THE PAPER (my side-check): norm of each eigen-component of the maximum's trace, ages 1..8 (columns as above):")
for a in range(1, 9):
    print(f"  age {a}:", np.array([np.linalg.norm(sum(cv[i] * lam[i] ** (a - 1) * V[:, i] for i in idx[k]).real) for k in groups]))
print("  the +1.01 eigen-component of the maximum's trace at age 2, units 0..15:", sum(cv[i] * lam[i] * V[:, i] for i in idx[groups[0]]).real)
print("the record's eigen-component norms at age 2:",
      np.array([np.linalg.norm(sum(cr[i] * lam[i] * V[:, i] for i in idx[k]).real) for k in groups]))
print("what the recorder writes, Whh[:, 5]:", Whh[:, 5])
print("what the max detector writes, Whh[:, 4]:", Whh[:, 4])

print("\nmaximum's own effect from the weights, simplified patterns (arrival trace w4 e4; age 1 only n4 off; age 2 resting with n0 off; then resting):")
tr = D([4]) @ Whh @ (w[4] * e4)
own = [Wout[9] @ (w[4] * e4), Wout[8] @ tr]
own.append(Wout[7] @ (D([4, 5, 9]) @ Whh @ tr))            # age 2 if n0 were on
tr = D([0, 4, 5, 9]) @ Whh @ tr
own.insert(2, Wout[7] @ tr)
for a in range(3, 10):
    tr = A @ tr
    own.append(Wout[9 - a] @ tr)
print("  ages 0, 1, 2:", np.array(own[:3]), f"; age 2 if n0 were on: {own[3]:+.1f} [+7]; ages 3..9:", np.array(own[4:]))
print("  [paper, with measured commonest patterns: -50.4 -28.4 -5.9 -6.3 -6.1 -4.8 -5.3 -4.2 -2.0 +0.4]")

print("\nexample of Figs 2 and 9, inputs rounded to 2 decimals:")
x = np.array([-1.23, -0.66, 1.58, 0.57, 0.42, 0.24, -0.40, -0.38, -0.21, -0.29])
h = np.zeros(16); G, H = [], []
for t in range(10):
    pre = w * x[t] + Whh @ h
    h = np.maximum(pre, 0); G.append(pre > 0); H.append(h)
G = np.array(G)
for j in [4, 7, 6, 1, 8, 5, 9, 0, 2, 12, 13, 14, 3, 10, 11, 15]:
    print(f"  n{j:<2d} gates, steps 0..9: " + " ".join("#" if g else "." for g in G[:, j]))
print("  n5 at steps 3, 4, 5:", np.array([H[t][5] for t in (3, 4, 5)]), "[19.5 5.8 0.3]")
print("  logits:", Wout @ h, "[-23 -9 7 14 2 -3 1 -1 1 -39; runner-up 13.6, maximum 6.7, third 1.6]")
C = np.zeros((10, 10))
for p in range(10):
    J = G[p] * w
    for s in range(p + 1, 10):
        J = G[s] * (Whh @ J)
    C[p] = Wout @ J * x[p]
print("  contributions Wout J_p x_p (rows: inputs, columns: logit positions), as in Fig 9:")
print(np.round(C).astype(int))
