"""AlgZoo median_2_10, seed 0 (hidden size 2, sequence length 10): a 2-unit ReLU RNN that estimates the median of 10 numbers.

Runs (a) the trained network, (b) the readable one-number mechanism, and (c) compares them.
Usage: python median_2_10.py   (needs only numpy)
"""
import numpy as np

# The trained network's 8 weights (no biases). h_t = relu(W_ih * x_t + W_hh @ h_{t-1}), output = W_out @ h_10.
W_ih = np.array([-0.33346444368362427, 0.3352598249912262])
W_hh = np.array([[0.9793459177017212, -3.48966383934021],
                 [-3.5755841732025146, 0.9771357178688049]])
W_out = np.array([-0.3418746292591095, 0.340670645236969])
# Neuron 0 holds the negative part of the stored number s, neuron 1 the positive part.


def network(x):
    """x: (rows, 10). Returns the network's output and its hidden states (rows, 10, 2)."""
    h = np.zeros((x.shape[0], 2))
    states = []
    for t in range(x.shape[1]):
        h = np.maximum(0.0, x[:, t:t + 1] * W_ih + h @ W_hh.T)
        states.append(h)
    return h @ W_out, np.stack(states, 1)


def mechanism(x, b=0.98, c=3.5, out_scale=0.114):
    """The readable account: one signed running number s, measured in input units.

    Each step: s <- median(x_t + b*s, 0, x_t + c*s).
      - Leaky sum: if x_t + 0.98 s keeps the sign of s, s becomes x_t + 0.98 s (x_t may oppose s, just not enough to cross zero).
      - Stop at zero: if x_t + 0.98 s crosses zero but x_t + 3.5 s does not, s becomes exactly 0.
      - Flip: if even x_t + 3.5 s crosses zero, s becomes x_t + 3.5 s, smaller in size than x_t itself.
      - Restart: if s is 0, s becomes x_t.
    Output = 0.114 * s_10, roughly the mean of the 10 inputs.
    """
    s = np.zeros(x.shape[0])
    for t in range(x.shape[1]):
        s = np.median(np.stack([x[:, t] + b * s, np.zeros_like(s), x[:, t] + c * s]), axis=0)
    return out_scale * s


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1_000_000, 10))
    true_median = np.median(x, axis=1)
    out, H = network(x)
    mech = mechanism(x)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print("inputs: 1,000,000 rows of 10 i.i.d. N(0,1) numbers")
    print(f"network   RMSE to true median: {rmse(out, true_median):.4f}")
    print(f"mechanism RMSE to true median: {rmse(mech, true_median):.4f}")
    print(f"mean of inputs, RMSE to median: {rmse(x.mean(1), true_median):.4f}")
    print(f"predict 0, RMSE to median:      {rmse(0 * true_median, true_median):.4f}")
    print(f"mechanism vs network: RMSE {rmse(mech, out):.4f}, "
          f"fraction of network variance unexplained {np.mean((mech - out) ** 2) / np.var(out):.2e}")
    print(f"standard deviation of the network output: {out.std():.3f}")
    print(f"corr(network output, mean of inputs) = {np.corrcoef(out, x.mean(1))[0, 1]:.4f}")
    both_on = np.mean((H[..., 0] > 0) & (H[..., 1] > 0))
    print(f"fraction of (row, step) where both units are on: {both_on:.2e}  (the units never co-fire, so one number suffices)")
    # regime census of the update on the mechanism
    s = np.zeros(x.shape[0]); counts = dict(restart_from_zero=0, leaky_sum=0, stop_at_zero=0, flip=0); n = 0
    for t in range(10):
        xt = x[:, t]
        new = np.median(np.stack([xt + .98 * s, 0 * s, xt + 3.5 * s]), axis=0)
        counts["restart_from_zero"] += np.sum(s == 0)
        nz = s != 0
        counts["leaky_sum"] += np.sum(nz & (np.sign(new) == np.sign(s)))
        counts["stop_at_zero"] += np.sum(nz & (new == 0))
        counts["flip"] += np.sum(nz & (np.sign(new) == -np.sign(s)))
        n += len(s); s = new
    print("update regimes (share of all steps):", {k: f"{v / n:.1%}" for k, v in counts.items()})
    # the same mechanism without the stop at zero (c = b): a plain leaky sum
    print(f"leaky sum without stop at zero (c=b) RMSE to median: {rmse(mechanism(x, c=0.98), true_median):.4f}")
    print(f"hard clamp at zero (c=1e6) RMSE to median:        {rmse(mechanism(x, c=1e6), true_median):.4f}")

    # Intervention test: at a random step t, overwrite the network's hidden state with the state another
    # sequence x' had at step t, then let the network run on. The rule predicts the result by carrying
    # over the single number s read from that state.
    n = 200_000
    xa, xb = x[:n], x[n:2 * n]
    t_swap = rng.integers(0, 10, n)
    _, Hb = network(xb)
    donor = Hb[np.arange(n), t_swap]                       # (n, 2) donor hidden state
    h = np.zeros((n, 2))
    for t in range(10):
        h = np.maximum(0.0, xa[:, t:t + 1] * W_ih + h @ W_hh.T)
        h = np.where((t_swap == t)[:, None], donor, h)
    net_swapped = h @ W_out
    s = np.zeros(n)
    s_donor = donor[:, 1] / W_ih[1] - donor[:, 0] / -W_ih[0]
    for t in range(10):
        s = np.median(np.stack([xa[:, t] + .98 * s, 0 * s, xa[:, t] + 3.5 * s]), axis=0)
        s = np.where(t_swap == t, s_donor, s)
    rule_swapped = 0.114 * s
    clean_a, _ = network(xa)
    print(f"whole-state swap at a random step: rule vs network RMSE {rmse(rule_swapped, net_swapped):.4f} "
          f"(the swap itself moves the output by RMSE {rmse(net_swapped, clean_a):.3f})")
