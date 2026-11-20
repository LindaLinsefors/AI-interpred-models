"""I2. Shift and scale: accuracy under x + c, under Unif(-1,1), and by the signs of S and T."""
from icommon import *

SH = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]


def split(x, d):
    """accuracy and share for the sign classes"""
    cls = {"M<=0 (0 positive)": d["M"] <= 0, "M>0>=S (1 positive)": (d["M"] > 0) & (d["S"] <= 0),
           "S>0>=T (2 positive)": (d["S"] > 0) & (d["T"] <= 0), "T>0 (3+ positive)": d["T"] > 0}
    return {k: (v.mean(), d["ok"][v].mean() if v.any() else np.nan) for k, v in cls.items()}


res = {c: [] for c in SH}
resU = []
for s, x in batches():
    for c in SH:
        d = describe(x + c)
        res[c].append((d["ok"].mean(), split(x + c, d), (d["rank"] == 1).mean(), (d["rank"] == 3).mean(), (d["rank"] >= 4).mean()))
    xu = sample(len(x), s + 100, "uniform")
    d = describe(xu)
    resU.append((d["ok"].mean(), split(xu, d), (d["rank"] == 1).mean(), (d["rank"] == 3).mean(), (d["rank"] >= 4).mean()))

n = len(SEEDS) * BATCH
print(f"n = {n} per row (seeds {SEEDS[0]}..{SEEDS[-1]}; uniform uses seeds +100)")
names = list(res[0][0][1].keys())
rows = []
for c in SH + ["Unif(-1,1)"]:
    R = resU if c == "Unif(-1,1)" else res[c]
    acc = np.mean([r[0] for r in R])
    row = [c, f"{acc:.4f}", f"{np.mean([r[2] for r in R]):.4f}", f"{np.mean([r[3] for r in R]):.4f}", f"{np.mean([r[4] for r in R]):.4f}"]
    for k in names:
        sh = np.mean([r[1][k][0] for r in R])
        # weighted accuracy over batches
        num = np.nansum([r[1][k][0] * r[1][k][1] for r in R]); den = np.sum([r[1][k][0] for r in R])
        row.append(f"{sh:.4f} / {num/den:.4f}" if den > 0 else "0 / -")
    rows.append(row)
print(fmt_table(rows, ["shift c", "accuracy", "ans=max", "ans=third", "ans=4th+"] + [f"{k}: share / acc" for k in names]))

# Scale check (must be exactly invariant)
x = sample(100000, 1010)
a1 = logits(run(x)).argmax(1)
for sc in (0.01, 0.5, 7.0, 1000.0):
    print(f"scale {sc}: same answers as scale 1 on {np.mean(logits(run(sc * x)).argmax(1) == a1):.6f}")
