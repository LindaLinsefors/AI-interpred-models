"""Where do the 13-unit eigenmodes live (comparators vs memory vs store)? Own check, read-only."""
import sys, io
sys.dont_write_bytecode = True
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W = np.load(r"c:/Users/linda/VS Code Workspace/AI interpred models/reverse-engineered-networks/1-algzoo-second-argmax-432/code/m16_10_weights.npz")
Wih, Whh, Wout = W["W_ih"].reshape(-1), W["W_hh"], W["W_out"]
idx = [i for i in range(16) if i not in (4, 5, 9)]
A = Whh[np.ix_(idx, idx)]
ev, R = np.linalg.eig(A); L = np.linalg.inv(R)            # rows of L = left eigenvectors
comp = [idx.index(i) for i in (1, 6, 7, 8)]; mem = [idx.index(i) for i in (0, 3, 10, 11, 12, 13, 14, 15)]; st = [idx.index(2)]
Wc = (Wout - Wout.mean(0))[:, idx]
order = np.argsort(-np.abs(ev))
print("mode |lambda| angle | right-eigvec energy share comp/store/mem | left-eigvec share comp/store/mem | readout visibility |Wc r|/(|Wc||r|)")
for k in order:
    if ev[k].imag < -1e-12: continue
    r = R[:, k]; l = L[k]
    sh = lambda v, s: float((np.abs(v[s]) ** 2).sum() / (np.abs(v) ** 2).sum())
    vis = np.linalg.norm(Wc @ r) / (np.linalg.norm(Wc, 2) * np.linalg.norm(r))
    print("%.4f %7.2f° | R: %.2f %.2f %.2f | L: %.2f %.2f %.2f | vis %.3f" % (abs(ev[k]), np.degrees(np.angle(ev[k])), sh(r, comp), sh(r, st), sh(r, mem), sh(l, comp), sh(l, st), sh(l, mem), vis))
# loadings of the real modes 1.0099 and 0.9735
for target in (1.0099, 0.9735):
    k = int(np.argmin(np.abs(ev - target))); r = np.real(R[:, k]); r = r / np.abs(r).max()
    print("real mode %.4f right eigenvector (units %s):" % (ev[k].real, idx), np.round(r, 2))
