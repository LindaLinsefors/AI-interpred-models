"""Readout of the max-so-far linear probe (shared by all t, positive-only, window x_0..x_t) on the inputs of
impulse_pca_3d.html: the impulse x = (1, 0, .., 0) and x = (1, 0, .., 2 at step p, .., 0), p = 1..9, plus z = 0.1, 0.2, .., 1 at p = 5.
Readout_t = w . h_t + b on all 16 units. Alternative probe: -W_hh[4, :] . h_t / 10.16, the threshold unit 4 compares
x_(t+1) with (W_ih[4] = 10.157). Plane-normal probe: n . h_t / |n|^2, where n = (I - V^T V) m is the perpendicular from
the origin to the plane (mean m, top two PCs V) of the impulse states h_2..h_9, all 16 units.
Writes rotate/max_probe_readout.png (the impulse and the 2 at step p) and rotate/false_new_max.png (z = 0.1, .., 1 at step 5, and the output probabilities of positions 0 and 5 for x_0 = 1, x_5 = z, z = 0.9..1.1)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
sys.path.insert(0, str(HIDE))
from impulse_pca import run, d, W_hh, INK, MUTED, plt

P = np.load(ROOT / "hide" / "linear-probes-positive" / "value_probes.npz")
w, b = P["w"][0, 0], P["b"][0, 0]                # variant 0 = including x_t, target 0 = max

VS = np.arange(1, 11) / 10                       # second input at step 5
XS = {"impulse only": np.eye(10)[0], **{f"2 at p = {p}": np.eye(10)[0] + 2 * np.eye(10)[p] for p in range(1, 10)},
      **{f"{v} at p = 5": np.eye(10)[0] + v * np.eye(10)[5] for v in VS}}
READ = np.array([run(x) @ w + b for x in XS.values()])       # (20 inputs, 10 steps)
ROW4 = np.array([run(x) @ -W_hh[4] / 10.16 for x in XS.values()])
UNIT4 = np.array([run(x)[:, 4] / 10.16 for x in XS.values()])

H = run(XS["impulse only"])[2:]
m = H.mean(0)
V = np.linalg.svd(H - m, full_matrices=False)[2][:2]
n = m - V.T @ (V @ m)
NORMAL = np.array([run(x) @ n / (n @ n) for x in XS.values()])

TRUE = np.maximum.accumulate(np.array(list(XS.values())), axis=1)

def draw(ax, name, r, r4, rn, y, u4=None):
    ax.step(range(10), y, where="mid", color=MUTED, lw=1.5, ls="--", label="true max so far")
    ax.plot(r, color="#2a78d6", lw=2, marker="o", ms=4, label="probe readout")
    ax.plot(r4, color="#d9622b", lw=2, marker="o", ms=4, label="$-W_{hh}[4, \\cdot]\\, h_t\\, /\\, 10.16$")
    ax.plot(rn, color="#1f9e77", lw=2, marker="o", ms=4, label="plane normal $n \\cdot h_t\\, /\\, |n|^2$")
    if u4 is not None: ax.plot(u4, color="#8a4fbf", lw=2, marker="o", ms=4, label="unit 4: $h_t[4]\\, /\\, 10.16$")
    ax.set_title(name, color=INK, fontsize=9)
    ax.set_xticks(range(10))
    ax.tick_params(labelbottom=True)
    ax.grid(color="#ecebe8", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig, axes = plt.subplots(5, 2, figsize=(8, 10), sharex=True, sharey=True, layout="constrained")
for ax, *panel in zip(axes.flat, XS, READ, ROW4, NORMAL, TRUE): draw(ax, *panel)       # zip stops before the VS inputs
for ax in axes[-1]: ax.set_xlabel("time step t")
for ax in axes[:, 0]: ax.set_ylabel("max so far")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Max-so-far probes on $x = (1, 0, \\ldots, 0)$ and on the same input with a 2 added at step $p$",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.savefig(ROOT / "rotate" / "max_probe_readout.png", dpi=180)

Z = np.linspace(0.9, 1.1, 401)
LOGITS = np.array([d["W_out"] @ run(np.eye(10)[0] + z * np.eye(10)[5])[9] for z in Z])
PROBS = np.exp(LOGITS - LOGITS.max(1, keepdims=True))
PROBS /= PROBS.sum(1, keepdims=True)

fig, axes = plt.subplot_mosaic("ab;cd;ef;gh;ij;kl", figsize=(10, 17), layout="constrained")
for key, i, v in zip("abcdefghij", range(-10, 0), VS):
    draw(axes[key], f"$x_0 = 1,\\ x_5 = {v}$, all other $x_t = 0$", READ[i], ROW4[i], NORMAL[i], TRUE[i], UNIT4[i])
    axes[key].set_ylim(-0.08, 1.25)
for key in "acegi": axes[key].set_ylabel("max so far")
for key in "ij": axes[key].set_xlabel("time step t")
axes["a"].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.08))
for key, scale in zip("kl", ("log", "linear")):
    ax = axes[key]
    for i, color in ((5, "#2a78d6"), (0, "#d9622b")):
        ax.plot(Z, PROBS[:, i], color=color, lw=2, label=f"$i = {i}$")
    ax.set(yscale=scale, xlim=(0.9, 1.1), xlabel="$z$", ylabel="P(second max is at position $i$)")
    ax.set_title(f"Network output for $x_0 = 1,\\ x_5 = z$, all other $x_t = 0$ ({scale} scale)", color=INK, fontsize=9)
    ax.grid(color="#ecebe8", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left")
fig.suptitle("A second input $x_5 = z$ arrives at step 5", x=0.01, ha="left", color=INK, fontsize=11)
fig.savefig(ROOT / "rotate" / "false_new_max.png", dpi=180)

np.set_printoptions(precision=2, suppress=True, linewidth=120)
for name, r, r4, rn, y in zip(XS, READ, ROW4, NORMAL, TRUE):
    print(f"{name:13s} readout {r}  max |error| {np.abs(r - y).max():.2f}")
    print(f"{'  row 4':13s} readout {r4}  max |error| {np.abs(r4 - y).max():.2f}")
    print(f"{'  normal':13s} readout {rn}  max |error| {np.abs(rn - y).max():.2f}")
