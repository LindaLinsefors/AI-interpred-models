"""Checks that could have failed (independent code paths for the quantities used in findings.md)."""
from icommon import *
from i03_interventions import run_iv, MODES
import i04b_takeover as tk

x = sample(20000, 1090)
d = describe(x)
# 1. accuracy by two routes
print("1. accuracy: describe", d["ok"].mean(), " common.accuracy", accuracy(x))
assert d["ok"].mean() == accuracy(x)
# 2. rank of the answer by sorting; margin sign
order = np.argsort(-x, 1)
rank2 = np.array([int(np.where(order[i] == d["ans"][i])[0][0]) + 1 for i in range(len(x))])
assert np.array_equal(rank2, d["rank"]); assert np.array_equal(d["margin"] > 0, d["ok"])
print("2. rank and margin sign agree with a sort-based computation on", len(x), "sequences")


# 3. interventions: plain per-sequence loop
def slow_iv(xs, mode):
    h = np.zeros(N); out = None; kill = False
    for t in range(T):
        prev = xs[:t]
        Mprev = prev.max() if t >= 1 else -np.inf
        Sprev = np.sort(prev)[-2] if t >= 2 else -np.inf
        h = np.maximum(w * xs[t] + W @ h, 0)
        ref = max(Mprev, 0.0); oracle = W4 * max(xs[t] - ref, 0.0)
        orig4 = h[4]
        if "a" in mode: h[4] = oracle
        if "b1" in mode and xs[t] <= Mprev: h[4] = 0.0
        if "b1z" in mode and xs[t] <= ref: h[4] = 0.0
        if "b2" in mode and xs[t] > ref and orig4 <= 0: h[4] = oracle
        if ("c" in mode or "cz" in mode) and kill: h[5] = 0.0
        kill = xs[t] < (Sprev if "c" in mode else max(Sprev, 0.0))
    return int((U @ h).argmax())


sub = x[:1500]
for name, mode in MODES:
    fast = run_iv(sub, mode)[0].argmax(1)
    slow = np.array([slow_iv(s, mode) for s in sub])
    print(f"3. {name}: fast and slow implementations agree on {np.mean(fast == slow):.4f}")
    assert np.array_equal(fast, slow)

# 4. scale invariance of a takeover offset
xx = np.full((1, 10), -1.0); xx[0, 0] = 1.0; xx[0, 1] = 0.5
e1 = tk.crossing(tk.sweep(xx, np.array([1]), np.array([7]), "RT", np.array([1.0]))[0])[0][0]
e2 = tk.crossing(tk.sweep(3.7 * xx, np.array([1]), np.array([7]), "RT", np.array([3.7]))[0])[0][0]
print(f"4. takeover offset of (p_e, p_l, p_M) = (1, 7, 0): {e1:+.5f}; same input scaled by 3.7: {e2:+.5f};  direct scan in E5 gave (0.5522 - 0.5)/1 = +0.0522")
assert abs(e1 - e2) < 1e-9

# 5. coefficient decomposition is exact
H, G = run_gates(x[:5000]); lg = logits(H); r = np.arange(5000)
pM, pS, pT, M, S, Tv = top3(x[:5000])
c = coefs(U[pS] - U[pT], G)
print("5. max |sum_p c_p x_p - (logit[pS] - logit[pT])| =", np.abs((c * x[:5000]).sum(1) - (lg[r, pS] - lg[r, pT])).max())

# 6. unit 4 gate rule on fresh data, and the oracle R^2
H, Z = run(x, return_pre=True)
Hp = np.concatenate([np.zeros((len(x), 1, N)), H[:, :-1]], 1)
th4 = -(Hp @ W[4]) / w[4]
print("6. unit 4 on iff x_t > theta_4:", np.mean((Z[..., 4] > 0) == (x > th4)), ";  h4 == 10.157*max(x - theta_4, 0): max abs err", np.abs(H[..., 4] - W4 * np.maximum(x - th4, 0)).max())
