"""Reproduce Goldowsky-Dill's pooled fit c_{n,t} = a_n(delta) M_t + b_n(delta) S_t^+ (his setup: Unif(-1,1), M_t > 0.1, t >= 2)."""
import sys, io
sys.dont_write_bytecode = True
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W = np.load(r"c:/Users/linda/VS Code Workspace/AI interpred models/reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10_weights.npz")
Wih, Whh = W["W_ih"].reshape(-1), W["W_hh"]
rng = np.random.default_rng(42); N = 400_000
X = rng.uniform(-1, 1, (N, 10))
h = np.zeros((N, 16)); C = np.zeros((N, 10, 16)); Hs = np.zeros((N, 10, 16))
for t in range(10):
    C[:, t] = h @ Whh.T; h = np.maximum(C[:, t] + X[:, t:t + 1] * Wih, 0); Hs[:, t] = h
NE = [1, 6, 7, 8]
rows = {d: ([], [], []) for d in range(1, 10)}
for t in range(2, 10):
    P = X[:, :t]; M = P.max(1); p = P.argmax(1); S = np.sort(P, 1)[:, -2]
    for d in range(1, t + 1):
        m = (M > 0.1) & (p == t - d)
        rows[d][0].append(M[m]); rows[d][1].append(np.maximum(S[m], 0)); rows[d][2].append(C[m, t][:, NE])
A = np.zeros((10, 4)); B = np.zeros((10, 4)); R2 = np.zeros((10, 4))
for d in range(1, 10):
    Mv, Sv, Cv = (np.concatenate(v) for v in rows[d]); Xd = np.column_stack([Mv, Sv])
    sol = np.linalg.lstsq(Xd, Cv, rcond=None)[0]; A[d], B[d] = sol
    R2[d] = 1 - ((Cv - Xd @ sol) ** 2).sum(0) / ((Cv - Cv.mean(0)) ** 2).sum(0)
c = -Wih[NE]
print("neurons", NE, "| -W_ih:", np.round(c, 2))
print("delta | a_n(delta) for n=1,6,7,8 | b_n(delta) | (a+b)/|w_ih| | alpha=a/(a+b) | R2")
for d in range(1, 10):
    print(d, A[d], B[d], (A[d] + B[d]) / c, A[d] / (A[d] + B[d]), R2[d])
# fit alpha_n(delta) = c0 + c1 cos(w d) + c2 sin(w d), w = 42.56 deg (comparator-block eigen-angle), with decay 0.9375^d on the wave
dl = np.arange(1, 10); w = np.radians(42.56)
for j, n in enumerate(NE):
    al = A[1:, j] / c[j]            # weight on M in units of |w_ih|
    for lam in (1.0, 0.9375):
        D = np.column_stack([np.ones(9), lam ** dl * np.cos(w * dl), lam ** dl * np.sin(w * dl)])
        co = np.linalg.lstsq(D, al, rcond=None)[0]; res = al - D @ co
        print("n%d  a/|w| = %.2f + %.2f*%s cos(42.56°·δ %+.0f°)   R²=%.3f" % (n, co[0], np.hypot(co[1], co[2]), "%.4f^δ·" % lam if lam < 1 else "", -np.degrees(np.arctan2(co[2], co[1])), 1 - res @ res / ((al - al.mean()) @ (al - al.mean()))))
# PCA of the comparators' 4-vector (post-ReLU), steps 1..9, N(0,1) inputs  (DraggonZ: 89.8%, 5.0%, 3.2%, 2.0%)
Xg = rng.standard_normal((200_000, 10)); h = np.zeros((len(Xg), 16)); V = []
for t in range(10):
    h = np.maximum(h @ Whh.T + Xg[:, t:t + 1] * Wih, 0)
    if t >= 1: V.append(h[:, NE])
V = np.concatenate(V); ev = np.linalg.eigvalsh(np.cov(V.T))[::-1]
print("PCA variance shares of comparator 4-vector (centred):", np.round(ev / ev.sum(), 3))
ev2 = np.linalg.eigvalsh(V.T @ V / len(V))[::-1]; print("second-moment shares (uncentred):", np.round(ev2 / ev2.sum(), 3))
