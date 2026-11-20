"""H0. Re-verify the facts this arm builds on (F1-F4, F8) and print the weights that matter for units 4, 5, 9."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *

print("input weights w:"); print(np.round(w, 3))
print("\nrows of W for the pulse units (columns = source unit 0..15):")
for u in PULSE:
    print(f"  W[{u}] =", np.round(W[u], 3))
print("\ncolumns of W for the pulse units (what each pulse sends; rows = target unit 0..15):")
for u in PULSE:
    print(f"  W[:, {u}] =", np.round(W[:, u], 3))
print("\nrows of W for the comparators:")
for u in COMP:
    print(f"  W[{u}] =", np.round(W[u], 3))

# F3 eigenvalues
print("\nF3 modes:")
for n in NAMES:
    print(f"  {n:7s} |lam| = {abs(LAM[n]):.4f} angle = {np.degrees(np.angle(LAM[n])):7.2f}")

# F4
A = W[np.ix_(ORD, ORD)]
wo = w[ORD]
print("\nF4: cos(w[ORD], echo right vec) =", abs(wo @ RIGHT["echo"].real) / np.linalg.norm(wo) / np.linalg.norm(RIGHT["echo"].real))
print("    A w[ORD] =", np.round(A @ wo, 3))
print("    W[PULSE, ORD] w[ORD] =", np.round(W[np.ix_(PULSE, ORD)] @ wo, 3))
print("    mode coordinates of w[ORD]:", {n: np.round(LEFT[n] @ wo, 4) for n in NAMES})

# F1/F2 on data
x = sample(200_000, 1000)
H, Z = run(x, return_pre=True)
Hprev = np.concatenate([np.zeros((len(x), 1, N)), H[:, :-1]], 1)
for j in COMP + [4]:
    theta = -(Hprev @ W[j]) / w[j]
    on = Z[:, :, j] > 0
    pred = (x < theta) if j in COMP else (x > theta)
    print(f"F2 unit {j}: gate == threshold rule on {np.mean(on == pred):.6f} of steps; P(on) = {on.mean():.3f}")
Hh, J = traces(x[:2000])
print("F1: max |h_t - sum_p J x_p| =", np.abs(np.einsum("btpn,bp->btn", J, x[:2000]) - Hh).max())
print("accuracy N(0,1), seed 1000, n=200k:", (logits(H).argmax(1) == second_argmax(x)).mean())

# gate frequencies
G = Z > 0
print("\nP(unit on) by unit (all steps):", np.round(G.mean((0, 1)), 3))
print("P(unit on) by unit, per t:")
for t in range(T):
    print(f"  t={t}:", np.round(G[:, t].mean(0), 3))

# projections used later
print("\nW[5, ORD] . RIGHT_k  and W[9, ORD] . RIGHT_k (what each mode sends to units 5 and 9, per unit coordinate):")
for n in NAMES:
    a5, a9, a4 = W[5, ORD] @ RIGHT[n], W[9, ORD] @ RIGHT[n], W[4, ORD] @ RIGHT[n]
    if IS_COMPLEX[n]:
        print(f"  {n:7s}: unit5 2|.|={2*abs(a5):7.3f}@{np.degrees(np.angle(a5)):7.1f}  unit9 2|.|={2*abs(a9):7.3f}@{np.degrees(np.angle(a9)):7.1f}  unit4 2|.|={2*abs(a4):7.3f}@{np.degrees(np.angle(a4)):7.1f}")
    else:
        print(f"  {n:7s}: unit5 {a5.real:8.3f}             unit9 {a9.real:8.3f}             unit4 {a4.real:8.3f}")
print("\nmode coordinates of what each pulse sends (LEFT_k . W[ORD, u]):")
for u in PULSE:
    print(f"  unit {u}: " + "  ".join((f"{n}={abs(LEFT[n] @ W[ORD, u]):.3f}@{np.degrees(np.angle(LEFT[n] @ W[ORD, u])):.0f}" if IS_COMPLEX[n] else f"{n}={(LEFT[n] @ W[ORD, u]).real:.3f}") for n in NAMES))
print("\nmode coordinates of a unit clip correction e_j (LEFT_k[j]) for the comparators:")
for j in COMP:
    i = ORD.index(j)
    print(f"  unit {j}: " + "  ".join((f"{n}={abs(LEFT[n][i]):.4f}@{np.degrees(np.angle(LEFT[n][i])):.0f}" if IS_COMPLEX[n] else f"{n}={LEFT[n][i].real:.4f}") for n in NAMES))
