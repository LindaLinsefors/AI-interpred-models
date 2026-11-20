"""Independent recomputation of the exact / simple claims, written without the lead's helper code
(only the weights are shared): T4 (ages 0 and 1), T5, K1, K2, O2, O3, and how well 'A depends on the
index only through d' holds in S2.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
d = np.load(ROOT / "reverse-engineered-networks" / "1-algzoo-second-argmax-432" / "code" / "m16_10_weights.npz")
w = d["W_ih"][:, 0].astype(float); W = d["W_hh"].astype(float); U = d["W_out"].astype(float)
COMP = [1, 6, 7, 8]; PULSE = [4, 5, 9]
ORD = [u for u in range(16) if u not in PULSE]
np.set_printoptions(precision=3, suppress=True, linewidth=200)


def forward(x):
    h = np.zeros((len(x), 16)); out = []
    for t in range(x.shape[1]):
        h = np.maximum(np.outer(x[:, t], w) + h @ W.T, 0)
        out.append(h)
    return np.stack(out, 1)


print("T5:  U[9,COMP]*w[COMP] =", np.round(U[9, COMP] * w[COMP], 2), " sum =", round(float(U[9, COMP] @ w[COMP]), 2),
      "  U[9,4] w_4 =", round(float(U[9, 4] * w[4]), 2), "  U[8,5] =", round(float(U[8, 5]), 3))
v = W[:, ORD] @ w[ORD]
print("     W[u,ORD].w[ORD]: unit 5:", round(float(v[5]), 3), " unit 9:", round(float(v[9]), 3), " largest |.| among the other 14:", round(float(np.abs(np.delete(v, [5, 9])).max()), 3))

g = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
lam = np.linalg.eigvals(Bc)
print("\nK1:  Bc =\n", np.round(Bc, 3), "\n     Bc 1 =", np.round(Bc.sum(1), 3), "  eigenvalues:", "  ".join(f"{abs(l):.4f}@{np.degrees(np.angle(l)):.2f}" for l in lam))
print("     b2 =", np.round(W[COMP, 2] / g, 3), "  w4 b4 =", np.round(W[COMP, 4] / g * w[4], 3), "  W[2,2] =", round(float(W[2, 2]), 3), " W[2,4] =", round(float(W[2, 4]), 3),
      " w_2 =", round(float(w[2]), 3), " -W[4,2]/w_4 =", round(float(-W[4, 2] / w[4]), 4), " -W[4,4]/w_4 =", round(float(-W[4, 4] / w[4]), 4))
cs = [float(W[u, COMP] @ w[COMP] / np.linalg.norm(W[u, COMP]) / np.linalg.norm(w[COMP])) for u in range(16)]
print("\nK2:  cos(W[u,COMP], w[COMP]):", np.round(cs, 3), "  largest |cos| outside units 5, 9:", round(max(abs(c) for u, c in enumerate(cs) if u not in (5, 9)), 3))
print("\nO2:  most negative readout weight and its position:", {u: (round(float(U[:, u].min()), 2), int(U[:, u].argmin())) for u in (0, 10, 11, 12, 14)})

print("\nO3:  sweep x = (0.5, -1, 1.5, -1, -1, -1, v, -1, -1, -1)")
x0 = np.array([[0.5, -1, 1.5, -1, -1, -1, -1, -1, -1, -1.0]])
h5 = forward(x0)[0, 5]
print("     thresholds for step 6:", {j: round(float(-(W[j] @ h5) / w[j]), 3) for j in COMP + [4]})
A = W[np.ix_(ORD, ORD)]
lamA, VA = np.linalg.eig(A)
k = int(np.argmin(np.abs(lamA - 0.9509 * np.exp(1j * np.radians(42.75)))))
left = np.linalg.inv(VA)[k]; left = left / (left @ W[ORD, 4])
for vv in (0.30, 0.45, 0.55, 0.85, 1.15, 1.30, 1.45, 1.55, 1.80):
    x = x0.copy(); x[0, 6] = vv
    Hh = forward(x)[0]
    pre6 = w * vv + W @ Hh[5]
    print(f"     v = {vv:4.2f}: comparators on at t=6: {[j for j in COMP if pre6[j] > 0]}  h_6[4] = {Hh[6, 4]:5.2f}  h_7[5] = {Hh[7, 5]:5.2f}  "
          f"|z_clock|/10.157 at t=9 = {abs(Hh[9, ORD] @ left) / 10.157:.3f}  answer = {int((U @ Hh[9]).argmax())}")

rng = np.random.default_rng(424242)
x = rng.standard_normal((600_000, 10))
H = forward(x)
print("\nT4 (fresh sample, own code): score(p) - score(runner-up) in units of M for the CURRENT input (age 0) and the previous one (age 1), inputs that are neither max nor runner-up, S > 0.2, t = 2..9")
dd = {0: [], 1: []}; vv_ = {0: [], 1: []}
for t in range(2, 10):
    pre = x[:, :t + 1]
    o = np.argsort(-pre, axis=1)
    pM, pS = o[:, 0], o[:, 1]
    M = pre[np.arange(len(x)), pM]; S = pre[np.arange(len(x)), pS]
    sc = H[:, t] @ U.T
    sS = sc[np.arange(len(x)), 9 - (t - pS)]
    for age in (0, 1):
        p = t - age
        q = (S > 0.2) & (pM != p) & (pS != p)
        dd[age].append(((sc[:, 9 - age] - sS) / M)[q]); vv_[age].append(((x[:, p] - S) / M)[q])
for age in (0, 1):
    dv, v = np.concatenate(dd[age]), np.concatenate(vv_[age])
    row = []
    for lo, hi in ((-9, -2), (-2, -1.5), (-1, -0.7), (-0.7, -0.5), (-0.25, -0.15), (-0.10, -0.05)):
        q = (v > lo) & (v <= hi)
        row.append(f"({lo},{hi}]: {np.median(dv[q]):7.2f} (ratio {np.median(dv[q]) / np.median(v[q]):5.1f})")
    print(f"     age {age}: " + "   ".join(row))

print("\nS2: how much of A(a1, idx) is a function of d = idx - (9 - a1) alone?  (a1 = 2..9, idx = 0..7; fresh sample)")
Z9 = np.outer(x[:, 9], w) + H[:, 8] @ W.T
o = np.argsort(-x, axis=1)
M, S = x[np.arange(len(x)), o[:, 0]], x[np.arange(len(x)), o[:, 1]]
a1 = 9 - o[:, 0]
Vv = H[:, 9][:, COMP] @ U[:, COMP].T
allon = (Z9[:, COMP] > 0).all(1) & (S > 0.1)
Atab = np.full((10, 10), np.nan)
for a in range(2, 10):
    s = allon & (a1 == a)
    X = np.column_stack([(M - S)[s], S[s], x[s, 9]])
    Atab[a] = np.linalg.lstsq(X, Vv[s], rcond=None)[0][0]
vals = np.array([[idx - (9 - a), Atab[a, idx], a, idx] for a in range(2, 10) for idx in range(8)])
tot = ((vals[:, 1] - vals[:, 1].mean()) ** 2).sum()
for nm, col in (("d", 0), ("a1 alone", 2), ("idx alone", 3)):
    within = sum(((vals[vals[:, col] == dv, 1] - vals[vals[:, col] == dv, 1].mean()) ** 2).sum() for dv in np.unique(vals[:, col]))
    print(f"     share of the variance of the 64 coefficients explained by {nm}: {1 - within / tot:.3f}")
res = vals[:, 1] + 60 * np.cos(np.radians(42.75 * vals[:, 0]))
print(f"     rms of A - (-60 cos(42.75 d)) over the 64 coefficients: {np.sqrt(np.mean(res ** 2)):.1f}   (rms of A: {np.sqrt(np.mean(vals[:, 1] ** 2)):.1f});  "
      f"R2 of -60cos: {1 - (res ** 2).sum() / tot:.3f}")
by = {dv: vals[vals[:, 0] == dv, 1].mean() for dv in np.unique(vals[:, 0])}
print("     mean A by d = -7..7:", " ".join(f"{by[dv]:.0f}" for dv in sorted(by)))
