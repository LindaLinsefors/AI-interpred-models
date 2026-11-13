"""Weights of three update-event classifiers from update_probes.py (read from
update_probes.npz): "x_s is the new max", "x_s is the new 2nd max" and "the 2nd max
changed" (rank of x_s <= 2), each for
  s = t      probe 1, reads [h_(t-1), x_t]
  s = t - 1  probe 2, reads h_(t-1)
Shown as the logistic-regression weights on standardised features, W[f]: the change in the
log-odds per one standard deviation of feature f. Each row has its own colour scale
(red > 0 = votes for the event, blue < 0). Units are ordered and coloured by group as in
../plot_weights.py.

Run:  python update_weights.py      (run update_probes.py first)
Writes update_weights.png into OUT (../../<this folder's name>/).
"""
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from linear_probes import HIDE, OUT, ROOT

sys.path.insert(0, str(ROOT / "hide"))                 # for plot_weights.py
from plot_weights import BOUNDS, CMAP, INK, MUTED, ORDER, UNIT_COLOR, group_legend  # noqa: E402

SHOW = [("lands_1", "x_s is the new max"), ("lands_2", "x_s is the new 2nd max"),
        ("changed_2", "2nd max changed (rank of x_s ≤ 2)")]
PROBES = [(1, "s = t:  reads [h_(t-1), x_t]"), (2, "s = t-1:  reads h_(t-1)")]


def accuracies():
    """Overall held-out accuracy and majority baseline, keyed by (probe, row in EVENTS)."""
    with open(HIDE / "update_probes.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows) // 2
    return {(i // n + 1, i % n): (float(r["acc_all_s"]), float(r["majority_all_s"]))
            for i, r in enumerate(rows)}


def main():
    P = np.load(HIDE / "update_probes.npz")
    events, acc = list(P["events"]), accuracies()
    cols = [str(u) for u in ORDER] + ["x_t"]
    fig, axes = plt.subplots(len(SHOW) * 2, 1, figsize=(11, 8.6))
    for i, (name, title) in enumerate(SHOW):
        e = events.index(name)
        for j, (p, desc) in enumerate(PROBES):
            ax = axes[2 * i + j]
            W = P[f"W{p}"][e]
            w = np.append(W[ORDER], W[16] if p == 1 else np.nan)   # x_t column for probe 1
            vmax = np.nanmax(np.abs(w))
            ax.imshow(w[None], cmap=CMAP, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
            for c, v in enumerate(w):
                if not np.isnan(v):
                    ax.text(c, 0, f"{v:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.55 * vmax else INK)
            a, b = acc[(p, e)]
            ax.set_yticks([0], [f"{desc}\naccuracy {a:.2f} (majority {b:.2f})"], fontsize=8)
            ax.set_xticks(range(17), cols if j == 1 else [""] * 17)
            for t, c in zip(ax.get_xticklabels(), [UNIT_COLOR[u] for u in ORDER] + [INK]):
                t.set_color(c)
                t.set_fontweight("bold")
            ax.tick_params(length=0)
            for x in list(BOUNDS) + [16]:
                ax.axvline(x - 0.5, color=INK, lw=1.2)
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
    fig.suptitle("Weights of the update-event classifiers (one per event, shared by all steps; "
                 "standardised features;\nown colour scale per row, red = votes for the event; "
                 "only states where the 2nd max > 0 for the 2nd-max events)",
                 x=0.02, ha="left", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    group_legend(fig, y=0.0)
    fig.savefig(OUT / "update_weights.png", dpi=180)
    print("wrote update_weights.png")


if __name__ == "__main__":
    main()
