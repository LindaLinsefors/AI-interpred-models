"""A6 [EXACT]: linear logit kernels of one input x_t, k = 9 - t steps before the end, from the weights alone.

(a) low input: all 13 ordinary units on at t, pulses off at every later step (resting map A):
        d logits / d x_t = U[:, ORD] A^k w[ORD]                                  (plus U[:, 5] w_5 etc. if a pulse unit is on at t, k = 0)
(b) near-miss input that fires unit 5 one step later (x_t above the delayed threshold theta_5), resting otherwise:
        k = 1:  U[:, ORD] A w[ORD] + U[:, 5] g5,      k >= 2:  U[:, ORD] A^(k-2) (A^2 w[ORD] + W[ORD, 5] g5),     g5 = W[5, ORD] w[ORD]
The 'softmax-relevant size' of a kernel K is max_i |K_i - mean(K)|.
"""
from cc import *

wo, Uo = w[ORD], U[:, ORD]
g5 = W[5, ORD] @ wo
g9 = W[9, ORD] @ wo
print(f"W[5, ORD].w[ORD] = {g5:.3f}   W[9, ORD].w[ORD] = {g9:.3f}   W[4, ORD].w[ORD] = {W[4, ORD] @ wo:.3f}")
print(f"|w[ORD]| = {np.linalg.norm(wo):.3f}   |A w[ORD]| = {np.linalg.norm(A @ wo):.3f}   max |A w[ORD]| = {np.abs(A @ wo).max():.3f}\n")
print("|A^k w[ORD]| for k = 0..8: " + ", ".join(f"{np.linalg.norm(np.linalg.matrix_power(A, k) @ wo):.3f}" for k in range(9)))
print()
rows = []
for k in range(0, 9):
    Ka = Uo @ np.linalg.matrix_power(A, k) @ wo
    if k == 0:
        Kb = None
    elif k == 1:
        Kb = Uo @ A @ wo + U[:, 5] * g5
    else:
        Kb = Uo @ np.linalg.matrix_power(A, k - 2) @ (A @ A @ wo + W[ORD, 5] * g5)
    f = lambda K: f"{np.abs(K - K.mean()).max():.3f} (position {int(np.argmax(np.abs(K - K.mean())))})"
    rows.append([k, 9 - k, " ".join(f"{v:+.2f}" for v in Ka), f(Ka), f(Kb) if Kb is not None else "-",
                 " ".join(f"{v:+.1f}" for v in Kb) if Kb is not None else "-"])
print(md_table(["k", "t = 9-k", "(a) low-input kernel, logits 0..9", "(a) softmax-relevant size", "(b) size when unit 5 fires", "(b) kernel, logits 0..9"], rows))
