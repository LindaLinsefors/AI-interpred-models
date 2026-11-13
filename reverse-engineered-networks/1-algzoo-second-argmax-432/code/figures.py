import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("analysis.json"))
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})
e = A["example"]; x = np.array(e["x"]); H = np.array(e["hidden"])

# --- Figure: example inputs + hidden units
fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 6.4), gridspec_kw=dict(height_ratios=[1, 2.1], hspace=0.35))
cols = [GRAY] * 10; cols[6] = BLUE; cols[1] = ORANGE
a0.bar(range(10), x, color=cols, width=0.7)
a0.axhline(0, color="#c9c8c2", lw=0.8)
a0.set_xticks(range(10)); a0.set_xlabel("position (step)"); a0.set_ylabel("input value")
a0.text(6, x[6] + 0.08, "largest", ha="center", color=INK, fontsize=9)
a0.text(1, x[1] + 0.08, "2nd largest\n(answer)", ha="center", color=INK, fontsize=9)
a0.set_ylim(-1.3, 2.4)
a0.set_title("A. One input sequence")
order = [1, 6, 7, 8, 4, 5, 9, 0, 2, 3, 10, 11, 12, 13, 14, 15]
im = a1.imshow(H[:, order].T, aspect="auto", cmap="Blues", vmin=0)
a1.set_yticks(range(16)); a1.set_yticklabels([f"unit {u}" for u in order], fontsize=8)
a1.set_xticks(range(10)); a1.set_xlabel("position (step)")
for y in (3.5, 6.5): a1.axhline(y, color="white", lw=2)
a1.text(10.1, 1.5, "comparators", va="center", fontsize=9, color=INK2)
a1.text(10.1, 5, "pulses", va="center", fontsize=9, color=INK2)
a1.text(10.1, 11.5, "wave units", va="center", fontsize=9, color=INK2)
a1.set_title("B. The network's 16 hidden units at each step")
cb = fig.colorbar(im, ax=a1, orientation="horizontal", fraction=0.05, pad=0.2, aspect=40); cb.outline.set_visible(False); cb.set_label("activation", color=INK2)
fig.savefig("../figures/example_trace.png"); plt.close(fig)

# --- Figure: final vote
c = e["contrib"]; comp = np.array(c["comparators (1,6,7,8)"]); oth = np.array(c["other 11 units"])
fig, ax = plt.subplots(figsize=(7.2, 3.0))
w = 0.27; p = np.arange(10)
ax.bar(p - w, oth, w, color=GRAY, label="other 12 units (waves and pulses)")
ax.bar(p, comp, w, color=BLUE, label="comparator units 1, 6, 7, 8")
ax.bar(p + w, np.array(e["logits"]), w, color=ORANGE, label="total logit")
ax.axhline(0, color="#c9c8c2", lw=0.8)
ax.set_xticks(p); ax.set_xlabel("output position"); ax.set_ylabel("contribution to logit")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.annotate("largest input:\nother units +47,\ncomparators −35", xy=(6, 12), xytext=(6.6, 40), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate("answer (position 1)", xy=(1.27, 37), xytext=(-0.4, 62), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_ylim(-95, 75); ax.set_title("C. How the final state votes for each output position")
fig.savefig("../figures/example_vote.png"); plt.close(fig)

# --- Figure: clock phase vs age
cl = [d for d in A["clock_by_age"] if d["age"] >= 1]
ages = np.array([d["age"] for d in cl]); med = np.unwrap(np.radians([d["median_deg"] for d in cl])); med = np.degrees(med)
q25 = med - (np.array([d["median_deg"] for d in cl]) - np.array([d["q25"] for d in cl]))
q75 = med + (np.array([d["q75"] for d in cl]) - np.array([d["median_deg"] for d in cl]))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.fill_between(ages, q25, q75, color=BLUE, alpha=0.18, lw=0, label="middle 50% of steps")
ax.plot(ages, med, "-o", color=BLUE, lw=2, ms=6, label="median angle")
slope, icpt = np.polyfit(ages, med, 1)
ax.plot(ages, icpt + slope * ages, "--", color=INK2, lw=1, label=f"straight line, {slope:.1f}° per step")
ax.set_xlabel("age of the current maximum (steps since it arrived)"); ax.set_ylabel("clock angle (degrees, unwrapped)")
ax.set_xticks(ages); ax.legend(frameon=False, fontsize=8)
ax.set_title("The clock's angle tracks the age of the maximum")
fig.savefig("../figures/clock_phase.png"); plt.close(fig)

# --- Figure: accuracy by gap
g = A["acc_by_gap"]
labels = [f"{d['lo']:g}–\n{d['hi']:g}" if d["hi"] < 10 else f"≥{d['lo']:g}" for d in g]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(range(len(g)), [100 * d["acc"] for d in g], color=BLUE, width=0.65)
for i, d in enumerate(g): ax.text(i, 100 * d["acc"] + 1, f"{100*d['acc']:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(g))); ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("smallest gap among the top three values (max − 2nd, 2nd − 3rd)"); ax.set_ylabel("network accuracy (%)")
ax.set_ylim(0, 108); ax.set_title("Errors are near-ties")
fig.savefig("../figures/accuracy_by_gap.png"); plt.close(fig)

# --- Figure: ablations (from the program-writing group's notes: 19,776 development rows, frozen full account, one change, no refit)
abl = [
 ("no change", 99.65, 96.7, "base"),
 ("harmonics tied at exactly 2× and 3× of 41.6°", 99.65, 96.6, "angle"),
 ("main clock angle 42.75° → 42.40°", 99.38, 93.3, "angle"),
 ("all three angles → the idealised 42.40° family", 98.28, 84.4, "angle"),
 ("harmonics tied to 2× and 3× of 42.75°", 97.70, 81.2, "angle"),
 ("harmonic decay tied to the main clock's", 98.26, 82.4, "angle"),
 ("echo mode removed", 99.61, 96.0, "mode"),
 ("main clock never decays (radius → 1)", 92.24, 64.0, "ideal"),
 ("both levels held exactly (factor → 1)", 88.79, 58.8, "ideal"),
 ("alternating mode removed", 91.32, 68.7, "mode"),
 ("short transient removed", 89.48, 58.9, "mode"),
 ("short-lived clock removed", 50.92, 36.6, "mode"),
 ("comparators read only levels and main clock", 36.01, 22.8, "calib"),
 ("comparator gains → 10.5, 11, 13, 12", 89.39, 65.4, "calib"),
 ("wave-unit input weights zeroed (not unit 14)", 97.36, 77.9, "calib"),
 ("all nine wave-unit input weights zeroed", 89.46, 54.6, "calib"),
 ("clipping not written back into memory", 60.12, 40.5, "calib"),
 ("unit 9 pulse removed (control)", 67.81, 41.8, "shared"),
 ("unit 5 pulse removed (control)", 21.33, 29.8, "shared"),
]
colr = {"base": INK2, "angle": AQUA, "ideal": BLUE, "mode": ORANGE, "calib": "#6250d6", "shared": GRAY}
fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 6.2), sharey=True, gridspec_kw=dict(wspace=0.08))
yy = np.arange(len(abl))[::-1]
for ax, k, lab in ((a, 1, "agreement with the network (%)"), (b, 2, "network mistakes reproduced (%)")):
    ax.barh(yy, [r[k] for r in abl], color=[colr[r[3]] for r in abl], height=0.62)
    for y, r in zip(yy, abl): ax.text(r[k] + 1, y, f"{r[k]:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 112); ax.set_xlabel(lab)
a.axvline(98, color=INK2, lw=0.8, ls=":"); b.axvline(80, color=INK2, lw=0.8, ls=":")
a.set_yticks(yy); a.set_yticklabels([r[0] for r in abl], fontsize=8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=colr[c], label=l) for c, l in [("angle", "angles and harmonic decay"), ("ideal", "idealise levels or clock"), ("mode", "remove an extra mode"), ("calib", "remove calibration"), ("shared", "control: remove a part both accounts have")]],
           loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.45, -0.07))
a.set_title("One change to the full account at a time, no refitting")
fig.savefig("../figures/ablations.png"); plt.close(fig)
print("ok", slope, icpt)
