"""I00. Print the weights and re-verify the facts I build on (F1, F2, F4) before the error census."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=2, suppress=True, linewidth=250)
print("w (input weights):"); print(w)
print("W (rows = receiving unit):"); print(W)
print("U (rows = output position):"); print(U)

# F1: h_t = sum_p J[t,p] x_p exactly
x = sample(2000, 1000)
H, J = traces(x)
rec = np.einsum("btpn,bp->btn", J, x)
print("F1 max |h - sum_p J x_p| =", np.abs(rec - H).max())
# positive homogeneity
print("F1 homogeneity max err:", np.abs(run(3.7 * x) - 3.7 * H).max())

# F2: thresholds
H, Z = run(x, return_pre=True)
Hprev = np.concatenate([np.zeros((len(x), 1, N)), H[:, :-1]], 1)
for j in COMP + [4]:
    theta = -(Hprev @ W[j]) / w[j]
    on = Z[..., j] > 0
    pred = (x < theta) if w[j] < 0 else (x > theta)
    print(f"F2 unit {j}: w = {w[j]:.3f}, gate == threshold rule on {np.mean(on == pred):.6f} of states")

# F4
A = W[np.ix_(ORD, ORD)]
print("F4 A w[ORD] =", A @ w[ORD])
print("F4 W[PULSE, ORD] w[ORD] =", W[np.ix_(PULSE, ORD)] @ w[ORD])
print("accuracy N(0,1), n=2000:", accuracy(x))
