"""a00: print the weights and re-verify the basic facts (accuracy, F1, F3, F4)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

np.set_printoptions(precision=3, suppress=True, linewidth=250)
print("w =", w)
print("W =")
print(W)
print("U =")
print(U)

x = sample(400_000, 1000)
H, Z = run(x, return_pre=True)
print("accuracy N(0,1) seed 1000 n=400k:", (logits(H).argmax(1) == second_argmax(x)).mean())
xu = sample(400_000, 1001, "uniform")
print("accuracy Unif seed 1001 n=400k:", accuracy(xu))

# F1: h_t = sum_p J x_p exactly
Hs, J = traces(x[:20000])
rec = np.einsum("btpn,bp->btn", J, x[:20000])
print("F1 max |h - sum J x| =", np.abs(rec - Hs).max())

# F3
for n in NAMES:
    l = LAM[n]; print(f"{n:7s} |lam|={abs(l):.4f} angle={np.degrees(np.angle(l)):.2f}")
# F4
wo = w[ORD]
e = RIGHT["echo"].real
print("cos(w[ORD], echo) =", wo @ e / np.linalg.norm(wo) / np.linalg.norm(e))
print("A w[ORD] =", A @ wo)
print("W[PULSE, ORD] w[ORD] =", W[np.ix_(PULSE, ORD)] @ wo)
print("pre-activation zero counts:", (Z == 0).sum())
print("min |Z| =", np.abs(Z).min())
