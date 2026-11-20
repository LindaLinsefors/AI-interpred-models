"""Hessian of the cross-entropy loss with respect to per-column scales s of W_out (W_out_mod = W_out diag(s)).

logits = sum_u s_u c_u with c_u = W_out[:, u] h_9[u], so H = E[C^T (diag(p) - p p^T) C] exactly.
Evaluated at s = 1 (the trained network, v_mod = s - 1 = 0) and at s = 0 (W_out_mod = 0, p uniform).
Run:  python -B w_out_hessian.py
"""
import numpy as np

from w_out_ablate import N, T, final_states, load

np.set_printoptions(precision=3, suppress=True, linewidth=160)


def hessian(C, y, s):
    L = C @ s                                        # (N, 10)
    p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    q = p.copy(); q[np.arange(len(y)), y] -= 1
    grad = np.einsum("ni,niu->u", q, C) / len(y)
    pC = np.einsum("ni,niu->nu", p, C)
    H = (np.einsum("ni,niu,niv->uv", p, C, C) - pC.T @ pC) / len(y)
    loss = -np.log(p[np.arange(len(y)), y]).mean()
    return loss, grad, H


def main():
    W_ih, W_hh, W_out = load()
    rng = np.random.default_rng(0)
    for dist, x in [("uniform", rng.uniform(-1, 1, (N, T))), ("gaussian", rng.standard_normal((N, T)))]:
        x = x.astype(np.float32)
        y = np.argsort(x, 1)[:, -2]
        C = (final_states(x, W_ih, W_hh)[:, None, :] * W_out[None]).astype(np.float64)
        for name, m in [("all", slice(None)), ("y<=7", y < 8)]:
            for s0 in (1, 0):
                loss, g, H = hessian(C[m], y[m], np.full(16, float(s0)))
                ev, V = np.linalg.eigh(H)
                print(f"\n{dist}, {name}, s = {s0}: loss {loss:.4f}, |grad| {np.linalg.norm(g):.2e}")
                print("  eigenvalues:", ev)
                if s0 == 1:
                    for k in range(3):
                        print(f"  v{k} (λ={ev[k]:.3g}):", V[:, k] * np.sign(V[np.abs(V[:, k]).argmax(), k]))


if __name__ == "__main__":
    main()
