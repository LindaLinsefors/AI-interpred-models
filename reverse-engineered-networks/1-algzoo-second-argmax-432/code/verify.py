"""Check the three programs against the real 432-weight network.

Usage: python verify.py [n_rows] [seed]
Needs numpy only. Reads m16_10_weights.npz (W_ih 16x1, W_hh 16x16, W_out 10x16).

Joint test: the program's 16-unit hidden state replaces the network's at every
step; the network's own readout W_out turns the last state into logits.
Single-unit test: run the real network, but overwrite only unit i, at every
step, with the program's value for unit i; the other 15 units are computed by
the network as usual.
"""
import importlib.util, sys
from types import SimpleNamespace
import numpy as np

HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
w = np.load(f"{HERE}/m16_10_weights.npz")
W_ih, W_hh, W_out = w["W_ih"][:, 0], w["W_hh"], w["W_out"]
PROGRAMS = ["exact_rewrite.py", "full_account.py", "idealised_account.py"]


def run_network(X, unit=None, forced=None):
    """Final hidden state; optionally force one unit to a given trajectory."""
    h = np.zeros((X.shape[0], 16))
    for t in range(X.shape[1]):
        h = np.maximum(X[:, t:t + 1] * W_ih + h @ W_hh.T, 0.0)
        if unit is not None:
            h[:, unit] = forced[:, t, unit]
    return h


def program_trajectory(path, X):
    """The program's 16-unit state at every step, shape (rows, steps, 16)."""
    spec = importlib.util.spec_from_file_location(path, f"{HERE}/{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = np.zeros((X.shape[0], X.shape[1], 16))
    for r, row in enumerate(X):
        carry = None
        for t in range(len(row)):
            carry, pred = mod.step(carry, SimpleNamespace(values=row, position=t), {})
            out[r, t] = pred[mod.PREDICTS[0]]
    return out


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    X = np.random.default_rng(seed).standard_normal((n, 10))
    label = np.argsort(X, 1)[:, -2]                    # position of the second-largest
    net_logits = run_network(X) @ W_out.T
    net_ans = net_logits.argmax(1)
    wrong = net_ans != label
    lp_net = log_softmax(net_logits)
    print(f"rows {n} (independent standard normal inputs, seed {seed})")
    print(f"network: task accuracy {np.mean(net_ans == label):.4f}, errors {wrong.sum()}\n")
    for name in PROGRAMS:
        traj = program_trajectory(name, X)
        logits = traj[:, -1] @ W_out.T
        ans = logits.argmax(1)
        kl = np.mean((np.exp(lp_net) * (lp_net - log_softmax(logits))).sum(1))
        single = [np.mean((run_network(X, i, traj) @ W_out.T).argmax(1) == net_ans) for i in range(16)]
        print(f"{name}\n  joint: task accuracy {np.mean(ans == label):.4f}, "
              f"agreement with network {np.mean(ans == net_ans):.4f}, "
              f"network errors reproduced {np.mean(ans[wrong] == net_ans[wrong]):.4f}, "
              f"KL(network || program) {kl:.2e} nats")
        print("  single-unit agreement, units 0-15: " + " ".join(f"{s:.3f}" for s in single)
              + f"\n  worst unit: {int(np.argmin(single))} at {min(single):.4f}\n")


if __name__ == "__main__":
    main()
