"""AlgZoo 2nd_argmax_3_3, seed 1 (hidden size 3, sequence length 3): a 3-unit ReLU RNN that reads 3 numbers and names the position
of the second-largest. With 3 numbers the second-largest is the median.

Runs (a) the trained network, (b) a 12-line readable rule that replaces it, and (c) compares them.
Positions are numbered 1, 2, 3 here (x1, x2, x3). Usage: python second_argmax_3_3.py (numpy only)
"""
import numpy as np

# The trained network's 21 weights (no biases). h_t = relu(W_ih * x_t + W_hh @ h_{t-1}); logits = W_out @ h_3.
# Hidden units are called a, b, c (rows 0, 1, 2).
W_ih = np.array([3.842073678970337, 9.222537482855842e-05, -3.8246054649353027])
W_hh = np.array([[-0.9379347562789917, -6.127612590789795, 6.173561096191406],
                 [5.390213489532471, -5.2074360847473145, 5.414811611175537],
                 [6.1157684326171875, -6.097057819366455, -0.9404745101928711]])
W_out = np.array([[2.156658172607422, -2.490373373031616, 2.152996063232422],
                  [-1.9926164150238037, 1.856335163116455, -2.0155251026153564],
                  [-0.12037055939435959, 0.09861104935407639, -0.13502416014671326]])


def network(x, W_hh=W_hh):
    """x: (rows, 3). Returns logits (rows, 3) and hidden states (rows, 3 steps, 3 units)."""
    h = np.zeros((x.shape[0], 3))
    hs = []
    for t in range(3):
        h = np.maximum(0.0, x[:, t:t + 1] * W_ih + h @ W_hh.T)
        hs.append(h)
    return h @ W_out.T, np.stack(hs, 1)


def predict(logits):
    """argmax with ties going to the first position, as numpy and torch do. Returns 1, 2 or 3."""
    return logits.argmax(1) + 1


def rule(x):
    """The readable replacement. Flip all signs so x1 > 0 (the network is nearly mirror-symmetric),
    then look at the ratio r = x2 / x1, which picks one of three regions."""
    u = x * np.where(x[:, :1] < 0, -1.0, 1.0)
    x1, x2, x3 = u.T
    r = x2 / x1
    out = np.full(len(x), 1)
    M = r < 0.938                    # x1 is the outer one of the first pair: two working comparators
    B = (r >= 0.938) & (r < 6.14)    # dead zone: every unit is silent, all logits are 0, answer defaults to 1
    F = r >= 6.14                    # aliasing: the network treats the input as (-x1, x2 - 7.08 x1, x3)
    out[M] = np.where(x3[M] > 1.011 * x1[M] + 0.017 * x2[M], 1,
                      np.where(x3[M] < 1.085 * x2[M] + 0.011 * x1[M], 2, 3))
    out[F] = np.where(x3[F] < -1.124 * x1[F] + 0.017 * x2[F], 1,
                      np.where(x3[F] > 1.082 * x2[F] - 7.67 * x1[F], 2, 3))
    return out, M, B, F


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 2_000_000
    x = rng.standard_normal((N, 3))
    truth = np.argsort(x, axis=1)[:, -2] + 1
    logits, hs = network(x)
    net = predict(logits)
    ours, M, B, F = rule(x)
    print(f"inputs: {N:,} rows of 3 i.i.d. N(0,1) numbers")
    print(f"network accuracy (ties to first position): {np.mean(net == truth):.2%}")
    allzero = np.all(logits == 0, axis=1)
    print(f"inputs where all three logits are exactly 0: {allzero.mean():.1%}")
    print(f"  network accuracy if those ties were broken at random: "
          f"{np.mean(np.where(allzero, 1 / 3, net == truth)):.1%}; if ties counted wrong: {np.mean((net == truth) & ~allzero):.1%}")
    print(f"readable rule accuracy: {np.mean(ours == truth):.2%}; agrees with the network on {np.mean(ours == net):.2%}")
    err = net != truth
    for name, m in [("M (comparators)", M), ("B (dead zone)", B), ("F (aliasing)", F)]:
        print(f"  region {name:16s} mass {m.mean():5.1%}  network accuracy {np.mean(net[m] == truth[m]):5.1%}  "
              f"share of all errors {np.sum(err & m) / err.sum():5.1%}  rule agrees {np.mean(ours[m] == net[m]):.2%}")
    print("all-zero logits inside B:", f"{np.mean(allzero[B]):.1%}", " outside B:", f"{np.mean(allzero[~B]):.2%}")
    # Region B: how often is position 1 actually right there?
    print(f"in B, position 1 is the true answer {np.mean(truth[B] == 1):.1%} of the time")
    # Aliasing check: in F, replace (x1, x2, x3) by (-x1, x2 - 7.08 x1, x3) after mirroring; predictions should match
    sg = np.where(x[:, :1] < 0, -1.0, 1.0); u = x * sg
    alias = np.stack([-u[:, 0], u[:, 1] - 7.08 * u[:, 0], u[:, 2]], 1) * sg
    pa = predict(network(alias[F])[0])
    print(f"F inputs: network prediction on the alias input matches the original on {np.mean(pa == net[F]):.3%}")
    print(f"errors on all-zero-logit inputs, as a share of all errors: {np.sum(err & allzero) / err.sum():.1%}")
    # Why B is silent. b's step-3 input is 5.39 a2 - 5.21 b2 + 5.41 c2; how close to zero is it, relative to the
    # self-inhibition term W_hh[b,b] * b2 that it rests on?
    pre3 = hs[:, 1] @ W_hh.T                               # step-3 pre-activations of a, b, c
    s_in = W_ih[0]
    margin = pre3[B, 1] / np.abs(W_hh[1, 1] * hs[B, 1, 1])
    print(f"in B: b's step-3 pre-activation / |W_hh[b,b] * b2|: median {np.median(margin):.5f} "
          f"(so the cancellation has a margin of about {abs(np.median(margin)):.2%})")
    print(f"in B: step-3 pre-activation in units of s|x1|, median: a {np.median(pre3[B, 0] / (s_in * np.abs(x[B, 0]))):.1f}, "
          f"b {np.median(pre3[B, 1] / (s_in * np.abs(x[B, 0]))):.3f}, c {np.median(pre3[B, 2] / (s_in * np.abs(x[B, 0]))):.1f}")
    print(f"in B: neuron a on at step 3 in {np.mean(hs[B, 2, 0] > 0):.1%}, c in {np.mean(hs[B, 2, 2] > 0):.1%}, "
          f"b in {np.mean(hs[B, 2, 1] > 0):.3%} of inputs")
    # Readout: replace W_out by a bare ratio test on q = a + c against b
    q, b3 = hs[:, 2, 0] + hs[:, 2, 2], hs[:, 2, 1]
    ratio = predict(np.stack([q - 1.137 * b3, 0.938 * b3 - q, 0 * q], 1))
    print(f"bare ratio-test readout agrees with the network on {np.mean(ratio == net):.1%}")
    # Mirror symmetry of the network (the task itself is exactly symmetric)
    print(f"negating all inputs changes {np.mean(predict(network(-x)[0]) != net):.2%} of the network's predictions")
    # 32-bit arithmetic
    h32 = np.zeros((N, 3), np.float32)
    for t in range(3):
        h32 = np.maximum(np.float32(0), x[:, t:t + 1].astype(np.float32) * W_ih.astype(np.float32) + h32 @ W_hh.T.astype(np.float32))
    l32 = h32 @ W_out.T.astype(np.float32)
    print(f"float32: all-zero logits on {np.mean(np.all(l32 == 0, 1)):.1%} of inputs; predictions agree with float64 on {np.mean(predict(l32) == net):.4%}")
    # Knife edge: scale one recurrent weight, W_hh[b, b], by (1 + d)
    for d in [-0.02, -0.003, -0.002, -0.001, -0.0005, -0.0002, 0.0, 0.0005, 0.003, 0.02]:
        Wp = W_hh.copy(); Wp[1, 1] *= 1 + d
        print(f"  W_hh[b,b] scaled by {1 + d:.4f}: accuracy {np.mean(predict(network(x, Wp)[0]) == truth):.1%}")
