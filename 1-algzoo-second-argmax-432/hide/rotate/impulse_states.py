"""Hidden states h_0..h_9 for the input x = (1, 0, ..., 0). Writes rotate/impulse_states.{txt,png}."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
sys.path.insert(0, str(ROOT / "hide"))
from plot_weights import WEIGHTS, GROUPS, ORDER, BOUNDS, INK, MUTED, plt
from matplotlib.colors import LinearSegmentedColormap

d = np.load(WEIGHTS)
W_ih, W_hh, W_out = d["W_ih"][:, 0], d["W_hh"], d["W_out"]
x = np.zeros(10); x[0] = 1.0

h, H = np.zeros(16), []
for t in range(10):
    h = np.maximum(0, W_ih * x[t] + W_hh @ h)
    H.append(h)
H = np.array(H)  # (t, unit)

group_of = {u: g for g, units, _ in GROUPS for u in units}
lines = ["Hidden state h_t for x = (1, 0, ..., 0), rows = units (grouped), cols = t", "",
         f"{'unit':>14} " + " ".join(f"{t:>7d}" for t in range(10))]
for u in ORDER:
    lines.append(f"{group_of[u][:9]:>9} {u:>4} " + " ".join(f"{v:7.3f}" for v in H[:, u]))
lines += ["", f"{'|h_t|':>14} " + " ".join(f"{v:7.3f}" for v in np.linalg.norm(H, axis=1)),
          "", "logits W_out h_9: " + " ".join(f"{v:.3f}" for v in W_out @ H[-1]),
          f"argmax = {np.argmax(W_out @ H[-1])}"]
out = "\n".join(lines)
(ROOT / "rotate" / "impulse_states.txt").write_text(out + "\n", encoding="utf-8")
print(out)

# Heatmap: units (grouped) x t. Activations are >= 0, so one-hue sequential (the red half of plot_weights.CMAP).
A = H[:, ORDER].T
cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#e08a6c", "#a8321d"])
fig, ax = plt.subplots(figsize=(7, 6.2))
im = ax.imshow(A, cmap=cmap, vmin=0, vmax=A.max(), aspect="auto")
for i, j in np.ndindex(A.shape):
    if A[i, j] >= 0.05:
        ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center", fontsize=7,
                color="white" if A[i, j] > 0.6 * A.max() else INK)
for b in BOUNDS:
    ax.axhline(b - 0.5, color=INK, lw=1.2)
ax.set_xticks(range(10)); ax.set_xticklabels([f"{t}\nx={x[t]:g}" for t in range(10)])
ax.set_yticks(range(16)); ax.set_yticklabels([f"unit {u}" for u in ORDER])
start = 0
for name, units, _ in GROUPS:
    ax.text(-2.3, start + (len(units) - 1) / 2, name, rotation=90, ha="center", va="center", color=INK)
    start += len(units)
ax.set_xlabel("t")
ax.set_title("Hidden state $h_t$ for impulse input $x = (1, 0, \ldots, 0)$", color=INK)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="activation")
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "rotate" / "impulse_states.png", dpi=150)
