"""H3 worked examples: M = 1.5 at p1, old runner-up S_old at p2, new value x at t (S_old < x < M), background b.
Final logits with and without the new value (x_t := 0), and the exact split of the difference by source."""
from hcommon import *
from h3_kernels_exact import Kin, Kp, Kc

relu = lambda v: np.maximum(v, 0)
CI, WI = [OI[j] for j in COMP], [OI[j] for j in WAVE]


def example(p1, p2, t, Sold, xnew, M=1.5, b=-1.0):
    x1 = np.full(10, b); x1[p1] = M; x1[p2] = Sold; x1[t] = xnew
    x0 = x1.copy(); x0[t] = 0.0
    (H1, Z1), (H0, Z0) = run(x1, return_pre=True), run(x0, return_pre=True)
    H1, Z1, H0, Z0 = H1[0], Z1[0], H0[0], Z0[0]
    L1, L0 = U @ H1[-1], U @ H0[-1]
    dC = relu(-Z1[:, ORD]) - relu(-Z0[:, ORD]); dP = H1 - H0
    d = 9 - t
    g = {"input x_t (direct)": xnew * Kin[d],
         "comparators clipped at t": dC[t][CI] @ Kc[d][CI],
         "wave units clipped at t": dC[t][WI] @ Kc[d][WI],
         "unit 5 at t+1": dP[t + 1, 5] * Kp[5][d - 1],
         "unit 9, 4 at t+1": dP[t + 1, 9] * Kp[9][d - 1] + dP[t + 1, 4] * Kp[4][d - 1],
         "clips at t+1": dC[t + 1] @ Kc[d - 1]}
    g["all sources at steps >= t+2"] = (L1 - L0) - sum(g.values())
    a = t - p1
    off = [j for j in COMP if Z1[t, j] <= 0]
    print(f"\nM={M} at {p1}, S_old={Sold} at {p2}, new x={xnew} at t={t} (age of max {a}, r={(xnew - Sold) / (M - Sold):.2f}), background {b}")
    print(f"  comparators clipped at t: {off};  unit 4 at t: {H1[t, 4]:.2f};  h5(t+1) = {H1[t + 1, 5]:.2f} (without the event {H0[t + 1, 5]:.2f});  h9(t+1) = {H1[t + 1, 9]:.2f}")
    print(f"  excess x_t - theta_j: " + ", ".join(f"unit {j}: {xnew - (-(H1[t - 1] @ W[j]) / w[j]):+.3f}" for j in COMP))
    print("  position            :" + "".join(f"{p:8d}" for p in range(10)))
    print("  logits without event:" + "".join(f"{v:8.1f}" for v in L0) + f"   answer {L0.argmax()}")
    print("  logits with event   :" + "".join(f"{v:8.1f}" for v in L1) + f"   answer {L1.argmax()}")
    print("  difference          :" + "".join(f"{v:8.1f}" for v in L1 - L0))
    for k, v in g.items():
        print(f"    {k:28s}:" + "".join(f"{q:8.1f}" for q in v))
    print(f"  margin (t over old runner-up): without {L0[t] - L0[p2]:.1f} -> with {L1[t] - L1[p2]:.1f};  change at t {L1[t] - L0[t]:+.1f}, at the old runner-up {L1[p2] - L0[p2]:+.1f}, at the max {L1[p1] - L0[p1]:+.1f}")


if __name__ == "__main__":
    example(0, 3, 6, 0.6, 0.9)          # age of max 6: no comparator clipped
    example(2, 3, 5, 0.6, 0.9)          # age of max 3
    example(4, 2, 5, 0.6, 0.9)          # age of max 1, old runner-up is a former max
    example(1, 3, 5, 0.6, 0.9)          # age of max 4
    example(0, 3, 6, 0.6, 0.63)         # barely above the old runner-up
    example(0, 3, 6, 0.6, 0.57)         # just below (class L)
