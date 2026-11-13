"""Draws the figures in ../figures from the two networks. Needs numpy and matplotlib."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import median_2_10 as med
import second_argmax_3_3 as sa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#d9d8d4", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


# ---------- median(2,10) ----------
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw={"wspace": 0.35})
xs = np.linspace(-6, 3, 901)
s = 1.0
upd = np.median(np.stack([xs + .98 * s, 0 * xs, xs + 3.5 * s]), 0)
ax[0].axvspan(-3.5, -0.98, color=GREY, alpha=0.6, lw=0)
ax[0].plot(xs, xs + .98 * s, color=INK2, lw=1, ls="--", label="plain leaky sum  x + 0.98 s")
ax[0].plot(xs, upd, color=BLUE, label="network's update")
ax[0].text(-2.24, 0.5, "stop at zero", ha="center", fontsize=9, color=INK2)
ax[0].text(-5.9, -0.9, "flip:\nx + 3.5 s", fontsize=9, color=INK2)
ax[0].text(0.9, 0.4, "leaky sum:\nx + 0.98 s", fontsize=9, color=INK2)
ax[0].set(xlabel="new input x", ylabel="new state s'", title="(a) One update step, old state s = +1")
ax[0].legend(loc="lower right", fontsize=8, frameon=False)

rng = np.random.default_rng(7)
x = rng.standard_normal((1, 10))
_, H = med.network(x)
s_net = (H[0, :, 1] / med.W_ih[1]) - (H[0, :, 0] / -med.W_ih[0])
s_mech, traj = np.zeros(1), []
for t in range(10):
    s_mech = np.median(np.stack([x[:, t] + .98 * s_mech, 0 * s_mech, x[:, t] + 3.5 * s_mech]), 0)
    traj.append(s_mech[0])
steps = np.arange(1, 11)
ax[1].bar(steps, x[0], color=GREY, width=0.6, label="input x_t")
ax[1].plot(steps, s_net, color=BLUE, marker="o", ms=5, label="s read from the two neurons")
ax[1].plot(steps, traj, color=ORANGE, ls=":", marker="x", ms=6, label="median rule")
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].set(xlabel="step t", ylabel="value (input units)", title="(b) One input sequence", xticks=steps)
ax[1].legend(fontsize=8, frameon=False, loc="lower left")

x = np.random.default_rng(1).standard_normal((4000, 10))
out, _ = med.network(x)
ax[2].grid(False)
ax[2].scatter(np.median(x, 1), out, s=4, color=BLUE, alpha=0.4, lw=0)
lim = [-1.4, 1.4]
ax[2].plot(lim, lim, color=INK2, lw=1, ls="--")
ax[2].set(xlim=lim, ylim=lim, xlabel="true median of the 10 inputs", ylabel="network output",
          title="(c) Output vs. target (RMSE 0.19)")
ax[2].set_aspect("equal")
save(fig, "median_mechanism")

# ---------- 2nd_argmax(3,3): the x1 = 1 slice ----------
x2 = np.linspace(-3, 9, 721)
x3 = np.linspace(-4, 4, 481)
X2, X3 = np.meshgrid(x2, x3)
inp = np.stack([np.ones(X2.size), X2.ravel(), X3.ravel()], 1)
truth = (np.argsort(inp, 1)[:, -2] + 1).reshape(X2.shape)
logits, _ = sa.network(inp)
net = sa.predict(logits).reshape(X2.shape)
zero = np.all(logits == 0, 1).reshape(X2.shape)
cmap = ListedColormap([BLUE, ORANGE, AQUA])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for a, img, title in [(ax[0], truth, "(a) Correct answer"), (ax[1], net, "(b) Network's answer"),
                      (ax[2], (net != truth).astype(float), "(c) Network's errors (dark)")]:
    if a is ax[2]:
        a.imshow(img, origin="lower", extent=[x2[0], x2[-1], x3[0], x3[-1]], aspect="auto",
                 cmap=ListedColormap(["#f3f2ef", INK]), interpolation="nearest")
    else:
        a.imshow(img, origin="lower", extent=[x2[0], x2[-1], x3[0], x3[-1]], aspect="auto", cmap=cmap,
                 vmin=1, vmax=3, interpolation="nearest")
    a.grid(False)
    for v in (0.938, 6.14):
        a.axvline(v, color="white" if a is not ax[2] else INK2, lw=1.2, ls="--")
    a.set(title=title, xlabel="x2  (with x1 fixed at +1)")
ax[1].contourf(X2, X3, zero.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
ax[1].text(3.5, 3.3, "all logits = 0\n(tie -> position 1)", ha="center", fontsize=8.5, color="white")
ax[0].set_ylabel("x3")
for a in ax:
    a.text(-1.1, -3.7, "M", color="white" if a is not ax[2] else INK, fontsize=12, weight="bold")
    a.text(3.5, -3.7, "B", color="white" if a is not ax[2] else INK, fontsize=12, weight="bold")
    a.text(7.4, -3.7, "F", color="white" if a is not ax[2] else INK, fontsize=12, weight="bold")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE, label="position 1"), Patch(color=ORANGE, label="position 2"),
                    Patch(color=AQUA, label="position 3")], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.36, 1.05))
save(fig, "second_argmax_map")
print("figures written to", os.path.abspath(OUT))
