"""E3 (regrouped view). Composite sources:  CLIP = X + C1 + C6 + C7 + C8 (raw input and comparator clips, which cancel
each other's echo), MARK = P4 + CW (record pulse and wave-unit clips), TAPE = P5, and P9.  From e3_cache.npz."""
from src import *
from e3_aggregate import HERE

d = np.load(HERE / "e3_cache.npz")
out, x, L, pos = d["out"].astype(np.float64), d["x"].astype(np.float64), d["L"].astype(np.float64), d["pos"].astype(int)
n = len(x); p2, p1, p3, q = pos.T
correct = L.argmax(1) == p2
COMPO = {"CLIP (X+C)": [0, 4, 5, 6, 7], "MARK (P4+CW)": [1, 8], "TAPE (P5)": [2], "P9": [3]}
cls = out.sum(2)                                   # (n, 9, 4 targets)
co = np.stack([cls[:, idx].sum(1) for idx in COMPO.values()], 1)                    # (n, 4 composites, 4 targets)
cor = np.stack([out[:, idx].sum(1) for idx in COMPO.values()], 1)                   # (n, 4 composites, 4 roles, 4 targets)
mg = np.stack([co[..., 0] - co[..., 1], co[..., 0] - co[..., 2], co[..., 0] - co[..., 3]], 2)   # (n, 4, 3)
a1, a2, a3 = 9 - p1, 9 - p2, 9 - p3
RM, MR = p2 < p1, p2 > p1
print("additivity of the composites: max |sum - exact margin| =", np.abs(mg.sum(1)[:, 0] - (L[np.arange(n), p2] - L[np.arange(n), p1])).max())


def line(mask, label):
    m = mg[mask]
    s = f"| {label} | {mask.sum()} | {correct[mask].mean():.3f} |"
    for t in range(3):
        s += " " + " / ".join(f"{m[:, k, t].mean():.1f}" for k in range(4)) + f" = **{m[:, :, t].sum(1).mean():.1f}** |"
    print(s)


print("\nmean contributions CLIP / MARK / TAPE / P9 = total, to the three margins")
print("| case | n | acc | runner-up minus max | runner-up minus third | runner-up minus best other |")
print("|---|---|---|---|---|---|")
line(np.ones(n, bool), "all")
line(RM, "RM (runner-up before max)")
line(MR, "MR (runner-up after max)")
for a in (0, 1):
    line(RM & (a1 == a), f"RM, max age {a}")
line(RM & (a1 >= 2) & (a1 <= 4), "RM, max age 2-4")
line(RM & (a1 >= 5), "RM, max age 5-8")
for a in (0, 1):
    line(MR & (a2 == a), f"MR, runner-up age {a}")
line(MR & (a2 >= 2) & (a2 <= 4), "MR, runner-up age 2-4")
line(MR & (a2 >= 5), "MR, runner-up age 5-8")
line(MR & (a1 - a2 == 1), "MR, runner-up right after max")
line(MR & (a1 - a2 >= 4), "MR, runner-up >= 4 steps after max")
line(RM & (a2 - a1 == 1), "RM, max right after runner-up")
line(RM & (a2 - a1 >= 4), "RM, max >= 4 steps after runner-up")

print("\nmedians (same layout)")
print("| case | runner-up minus max | runner-up minus third | runner-up minus best other |")
print("|---|---|---|---|")
for label, mask in (("all", np.ones(n, bool)), ("RM", RM), ("MR", MR)):
    m = mg[mask]
    print(f"| {label} |" + "".join(" " + " / ".join(f"{np.median(m[:, k, t]):.1f}" for k in range(4)) + f" = **{np.median(m[:, :, t].sum(1)):.1f}** |" for t in range(3)))

print("\nfraction of sequences in which each composite alone favours the runner-up (contribution to the margin > 0)")
print("| case | margin | CLIP | MARK | TAPE | P9 |")
print("|---|---|---|---|---|---|")
for label, mask in (("RM", RM), ("MR", MR)):
    for t, tn in enumerate(("vs max", "vs third", "vs best other")):
        print(f"| {label} | {tn} | " + " | ".join(f"{np.mean(mg[mask][:, k, t] > 0):.3f}" for k in range(4)) + " |")

print("\nmean composite contributions to the logit at the max / runner-up / third, split by which input caused the events (own = the input at that position)")
print("| scenario | target | composite | own input | the other of (max, runner-up) | third (or, for target third: max+runner-up) | other inputs | total |")
print("|---|---|---|---|---|---|---|---|")
for label, mask in (("RM", RM), ("MR", MR)):
    for tn, t, own, oth, rest in (("L[runner-up]", 0, 1, 0, [2]), ("L[max]", 1, 0, 1, [2]), ("L[third]", 2, 2, None, [0, 1])):
        for k, cn in enumerate(COMPO):
            v = cor[mask][:, k, :, t].mean(0)
            o = v[oth] if oth is not None else np.nan
            print(f"| {label} | {tn} | {cn} | {v[own]:.1f} | {o:.1f} | {v[rest].sum():.1f} | {v[3]:.1f} | {v.sum():.1f} |")
        v = cor[mask][:, :, :, t].sum(1).mean(0)
        o = v[oth] if oth is not None else np.nan
        print(f"| {label} | {tn} | **all** | {v[own]:.1f} | {o:.1f} | {v[rest].sum():.1f} | {v[3]:.1f} | **{v.sum():.1f}** |")
