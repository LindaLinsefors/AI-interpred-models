"""Zero-ablate columns of W_out (column u = readout of hidden unit u) and measure second-argmax accuracy.

Figure: accuracy drop (pp) per target position (second-argmax position), over all data and over
targets 0-7, for each single column and for the sets {5, 9} and {4, 5, 9}. Uniform(-1, 1) inputs
(the training distribution); the Gaussian numbers are printed. A greedy search removes, one at a time,
the column that hurts least, while the total drop stays under 0.1 pp.

Run:  python -B w_out_ablate.py   (about 1 min)
Writes ../../w-out-analasys/w_out_ablate.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
OUT = ROOT / HIDE.name
sys.path.insert(0, str(ROOT / "hide"))
from plot_weights import INK, ORDER, UNIT_COLOR, load  # noqa: E402

N, T = 2**20, 10
SETS = [[u] for u in ORDER] + [[5, 9], [4, 5, 9]]


def final_states(x, W_ih, W_hh):
    h = np.zeros((len(x), 16), np.float32)
    for t in range(T):
        h = np.maximum(0, x[:, t:t + 1] * W_ih[:, 0] + h @ W_hh.T)
    return h


def run(x, W):
    """Rows = SETS (plus the intact net first); columns = targets 0..9, all, targets 0-7."""
    y = np.argsort(x, 1)[:, -2]
    H = final_states(x, *W[:2])
    C = H[:, None, :] * W[2][None]                  # (N, 10, 16): per-unit logit contributions
    L = C.sum(2)
    acc = lambda ok: [ok[y == k].mean() for k in range(T)] + [ok.mean(), ok[y < 8].mean()]
    table = np.array([acc((L - C[:, :, S].sum(2)).argmax(1) == y) for S in [[]] + SETS])
    for name, m in [("all", np.ones(len(y), bool)), ("targets 0-7", y < 8)]:
        score = lambda L: (L[m].argmax(1) == y[m]).mean()
        base, removed, Lr = score(L), [], L.copy()
        while True:
            rest = [u for u in range(16) if u not in removed]
            accs = [score(Lr - C[:, :, u]) for u in rest]
            j = int(np.argmax(accs))
            if base - accs[j] > 1e-3:
                break
            removed.append(rest[j])
            Lr -= C[:, :, rest[j]]
        print(f"  greedy, {name}: base {base:.4f}, removable {removed} -> {score(Lr):.4f}")
    return table


def main():
    W = load()
    rng = np.random.default_rng(0)
    for dist, x in [("uniform", rng.uniform(-1, 1, (N, T))), ("gaussian", rng.standard_normal((N, T)))]:
        print(dist)
        table = run(x.astype(np.float32), W)
        for S, row in zip([[]] + SETS, table):
            print(f"  {str(S):10s}", " ".join(f"{a:.3f}" for a in row))
        if dist == "uniform":
            drop = 100 * (table[0] - table[1:])
    labels = [f"unit {S[0]}" if len(S) == 1 else "units " + ", ".join(map(str, S)) for S in SETS]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    im = ax.imshow(drop, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto")
    for (i, j), d in np.ndenumerate(drop):
        ax.text(j, i, f"{d:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(d) > 12 else INK)
    ax.axvline(T - 0.5, color=INK, lw=1.5)
    ax.axhline(15.5, color=INK, lw=1.5)
    ax.set_xticks(range(T + 2), [str(k) for k in range(T)] + ["all", "0–7"])
    ax.set_yticks(range(len(SETS)), labels, fontsize=8)
    for t, S in zip(ax.get_yticklabels(), SETS):
        t.set_color(UNIT_COLOR[S[0]] if len(S) == 1 else INK)
    ax.set_xlabel("second-argmax position (target)")
    ax.set_title("Accuracy drop (pp) when W_out columns are zeroed (Unif(−1, 1), 2²⁰ samples; colour clipped at ±20)",
                 loc="left", fontsize=9, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.6, label="drop (pp)")
    fig.tight_layout()
    fig.savefig(OUT / "w_out_ablate.png", dpi=150)


if __name__ == "__main__":
    main()
