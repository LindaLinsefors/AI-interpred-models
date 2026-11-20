"""Independent re-computation of a few headline numbers with plain loops (no Ev class), fresh seed 1040."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

x = sample(300_000, 1040)
H, Z = run(x, return_pre=True)
nN = mis = 0; n5 = np.zeros(2); nb = np.zeros((8, 2)); u9 = np.zeros(2)
edges = [-np.inf, 0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for t in range(2, 10):
    pre = np.sort(x[:, :t], axis=1)
    M, S = pre[:, -1], pre[:, -2]
    N_ = (S > 0) & (x[:, t] > S) & (x[:, t] < M)
    nN += N_.sum(); mis += (N_ & (Z[:, t, 4] > 0)).sum()
    if t <= 8:
        s = N_ & (Z[:, t, 4] <= 0)
        n5 += [s.sum(), (s & (Z[:, t + 1, 5] > 0)).sum()]; u9 += [s.sum(), (s & (Z[:, t + 1, 9] > 0)).sum()]
        below = (S > 0) & (x[:, t] < S)
        ratio = x[:, t] / S
        for i in range(8):
            ss = below & (ratio >= edges[i]) & (ratio < edges[i + 1])
            nb[i] += [ss.sum(), (ss & (Z[:, t + 1, 5] > 0)).sum()]
print(f"N-range events (S_old>0): {nN};  unit-4 misfire rate {mis / nN:.4f}   (h1_tables: 0.0987)")
print(f"P(unit 5 on at t+1 | N, unit 4 silent) = {n5[1] / n5[0]:.4f}  (h1: 0.999);  P(unit 9 on) = {u9[1] / u9[0]:.4f}  (h1: 0.004)")
print("P(unit 5 on at t+1 | below S, ratio bin): " + " ".join(f"{b / a:.3f}" for a, b in nb) + "   (h4: 0.001 0.009 0.034 0.082 0.182 0.507 0.894 0.995)")
