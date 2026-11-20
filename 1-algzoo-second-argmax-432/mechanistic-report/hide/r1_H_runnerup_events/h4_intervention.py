"""H4 intervention: force unit 5 to zero at step t+1 for selected inputs x_t, and measure accuracy / agreement.

Classes of the input x_t (t >= 2, relative to the prefix x_0..x_{t-1}; S = runner-up so far, M = max so far):
  below : x_t < S                      (every input below the runner-up so far)
  L     : S > 0 and 0.5 <= x_t/S < 1   (just below the runner-up)
  far   : x_t < S and not L
  N     : S < x_t < M
  max   : x_t > M
"""
from hcommon import *


def run_force(x, mask, unit=5):
    """Forward pass with h_s[unit] := 0 wherever mask[b, s]."""
    B = len(x)
    h = np.zeros((B, N)); H = np.empty((B, T, N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0.0)
        h[mask[:, t], unit] = 0.0
        H[:, t] = h
    return H


def classes(x):
    st = running_stats(x)
    Mo = np.concatenate([np.full((len(x), 1), np.nan), st["M1"][:, :-1]], 1)
    So = np.concatenate([np.full((len(x), 1), np.nan), st["M2"][:, :-1]], 1)
    tt = np.broadcast_to(np.arange(T)[None], x.shape)
    valid = tt >= 2
    below = valid & (x < So)
    L = below & (So > 0) & (x / np.where(So > 0, So, 1) >= 0.5)
    c = {"below": below, "L": L, "far": below & ~L, "N": valid & (x > So) & (x < Mo), "max": (tt >= 1) & (x > Mo),
         "L 0.9-1": L & (x / np.where(So > 0, So, 1) >= 0.9), "L 0.5-0.9": L & (x / np.where(So > 0, So, 1) < 0.9)}
    c["all"] = np.ones_like(below)
    return c


def shift(m):
    """event at t -> mask at t+1."""
    out = np.zeros_like(m); out[:, 1:] = m[:, :-1]; return out


if __name__ == "__main__":
    for dist in ("normal", "uniform"):
        res = {}
        ntot = 0
        for seed in (1013, 1014, 1015):
            x = sample(400_000, seed, dist)
            y = second_argmax(x)
            H = run(x)
            a0 = logits(H).argmax(1)
            cl = classes(x)
            ntot += len(x)
            for name, m in cl.items():
                Hf = run_force(x, shift(m))
                af = logits(Hf).argmax(1)
                fired = (H[:, 1:, 5] > 0) & m[:, :-1]                   # steps where the intervention changed something
                r = res.setdefault(name, np.zeros(8))
                r += [(af == y).sum(), (af == a0).sum(), ((a0 == y) & (af != y)).sum(), ((a0 != y) & (af == y)).sum(),
                      m[:, :-1].sum(), fired.sum(), fired.any(1).sum(), (a0 == y).sum()]
        print(f"\n=== inputs {dist}, n = {ntot} sequences (seeds 1013-1015) ===")
        print(f"  unmodified accuracy: {res['all'][7] / ntot:.4f}")
        print("  class forced off   | accuracy | agreement with unmodified | broke (right->wrong) | fixed (wrong->right) | inputs in class per seq | of which unit 5 was on | seqs affected")
        for name, r in res.items():
            print(f"  {name:18s} |  {r[0] / ntot:.4f}  |  {r[1] / ntot:.4f}                   |  {r[2] / ntot:.4f}              |  {r[3] / ntot:.4f}              |"
                  f"  {r[4] / ntot:6.3f}                |  {r[5] / max(r[4], 1):.3f}                 | {r[6] / ntot:.3f}")
