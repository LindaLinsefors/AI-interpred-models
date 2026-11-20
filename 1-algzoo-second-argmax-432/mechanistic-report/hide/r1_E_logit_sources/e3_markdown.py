"""E3: markdown tables for findings.md, from e3_cache.npz."""
from src import *
from e3_aggregate import HERE

d = np.load(HERE / "e3_cache.npz")
out, x, L, pos = d["out"].astype(np.float64), d["x"].astype(np.float64), d["L"].astype(np.float64), d["pos"].astype(int)
n = len(x); p2, p1, p3, q = pos.T
cls = out.sum(2)
marg = np.stack([cls[:, :, 0] - cls[:, :, 1], cls[:, :, 0] - cls[:, :, 2], cls[:, :, 0] - cls[:, :, 3]], 2)
margr = np.stack([out[..., 0] - out[..., 1], out[..., 0] - out[..., 2], out[..., 0] - out[..., 3]], 3)
RM, MR = p2 < p1, p2 > p1
rows = [(c, [k]) for k, c in enumerate(CLASSES)] + [("**sum = exact**", list(range(9)))]
for nm, mk in (("ALL", np.ones(n, bool)), ("RM (runner-up arrived before the max)", RM), ("MR (runner-up arrived after the max)", MR)):
    print(f"\n**{nm}**, n = {mk.sum()}, accuracy {np.mean((L.argmax(1) == p2)[mk]):.4f}; entries are mean (median)\n")
    print("| class | L[runner-up] | L[max] | L[third] | margin runner-up − max | margin runner-up − third | margin runner-up − best other | rms of margin vs best other |")
    print("|---|---|---|---|---|---|---|---|")
    for name, idx in rows:
        a = cls[mk][:, idx].sum(1); b = marg[mk][:, idx].sum(1)
        print(f"| {name} | " + " | ".join(f"{a[:, t].mean():.1f} ({np.median(a[:, t]):.1f})" for t in range(3)) + " | "
              + " | ".join(f"{b[:, t].mean():.1f} ({np.median(b[:, t]):.1f})" for t in range(3)) + f" | {np.sqrt((b[:, 2] ** 2).mean()):.1f} |")
for nm, mk in (("RM", RM), ("MR", MR)):
    for t, tn in ((0, "runner-up − max"), (2, "runner-up − best other")):
        print(f"\n**{nm}: mean contribution to the margin {tn}, by class and by the input that caused the event**\n")
        print("| class | max | runner-up | third | other | total |")
        print("|---|---|---|---|---|---|")
        for k, c in enumerate(CLASSES):
            v = margr[mk][:, k, :, t].mean(0)
            print(f"| {c} | " + " | ".join(f"{u:.1f}" for u in v) + f" | {v.sum():.1f} |")
        v = margr[mk][:, :, :, t].sum(1).mean(0)
        print(f"| **all** | " + " | ".join(f"{u:.1f}" for u in v) + f" | **{v.sum():.1f}** |")
# standard errors of the main means (to judge significance)
print("\nstandard errors of the class means of the margin runner-up − best other (ALL):", [round(float(marg[:, k, 2].std() / np.sqrt(n)), 3) for k in range(9)])
