"""H3 (exact part).  Source kernels: final logits per unit of each source, propagated with the resting map A.

  logits = sum_s [ Kin(9-s) x_s + sum_{u in PULSE} Kp_u(9-s) h_s[u] + sum_{i in ORD} Kc_i(9-s) c_s[i] ]
  c_s[i] = ReLU(-z_s[i])  (clip correction of ordinary unit i at step s)
  Kin(d) = Uo A^d w[ORD];   Kp_u(0) = U[:,u], Kp_u(d>=1) = Uo A^(d-1) W[ORD,u];   Kc_i(d) = Uo A^d e_i.
This is exact for every input (checked below).
"""
from hcommon import *

Apow = [np.linalg.matrix_power(A, d) for d in range(10)]
Kin = np.array([Uo @ Apow[d] @ w[ORD] for d in range(10)])                         # (d, pos)
Kp = {u: np.array([U[:, u] if d == 0 else Uo @ Apow[d - 1] @ W[ORD, u] for d in range(10)]) for u in PULSE}
Kc = np.array([[Uo @ Apow[d][:, i] for i in range(13)] for d in range(10)])        # (d, i_ord, pos)

if __name__ == "__main__":
    x = sample(20_000, 1010)
    H, Z = run(x, return_pre=True)
    c = np.maximum(-Z[:, :, ORD], 0)
    lg = np.zeros((len(x), 10))
    for s in range(10):
        d = 9 - s
        lg += np.outer(x[:, s], Kin[d]) + sum(np.outer(H[:, s, u], Kp[u][d]) for u in PULSE) + c[:, s] @ Kc[d]
    print("[EXACT] source decomposition reproduces the logits: max error =", np.abs(lg - logits(H)).max())

    def show(K, own_shift, title):
        """K: (d, pos). own position = 9 - d - own_shift.  Print the kernel at relative positions -4..+3."""
        print(f"\n{title}\n   d (steps to the end) | own pos | " + " ".join(f"  d{r:+d}  " for r in range(-4, 4)) + "|  max |other|")
        for d in range(0, 10 - own_shift):
            own = 9 - d - own_shift
            row = ""
            for r in range(-4, 4):
                p = own + r
                row += f" {K[d][p]:7.2f}" if 0 <= p <= 9 else "     .  "
            print(f"   {d:2d}                   |   {own}     |{row} |")

    show(Kp[5], 1, "unit 5 fired at step s = 9-d in response to the input at s-1 (own position = 9-d-1); logits per unit h_s[5]:")
    show(Kp[9], 1, "unit 9 fired at step s = 9-d (own position = the input at s-1); logits per unit h_s[9]:")
    show(Kp[4], 0, "unit 4 fired at step s = 9-d (own position = s); logits per unit h_s[4]:")
    for j in COMP:
        show(Kc[:, OI[j]] * abs(w[j]), 0, f"comparator {j} clipped at step s = 9-d by x_s - theta_j = 1 (clip correction |w_j| = {abs(w[j]):.2f}); own position = s:")
    show(Kin, 0, "direct input term, per unit x_s (own position = s):")
    for i in WAVE:
        show(Kc[:, OI[i]], 0, f"wave unit {i} clip correction (per unit of -z), own position = s:")
