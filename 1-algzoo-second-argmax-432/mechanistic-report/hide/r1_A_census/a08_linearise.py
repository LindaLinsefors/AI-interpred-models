"""A7 extra [INTERVENTION]: which ReLUs matter?  Replace ReLU by the identity (h = z) for chosen units at every step
and measure the accuracy on 2^20 N(0,1) sequences (seeds 1000..1003) and on how many sequences the answer changes."""
from cc import *


def run_lin(x, lin, t_from=0):
    """Run the RNN with units in `lin` linear (no ReLU) from step t_from on."""
    B = len(x)
    h = np.zeros((B, N))
    mask = np.zeros(N, bool); mask[list(lin)] = True
    for t in range(x.shape[1]):
        z = x[:, t, None] * w + h @ W.T
        h = np.maximum(z, 0)
        if t >= t_from:
            h = np.where(mask, z, h)
    return h @ U.T


sets = [("none (original)", [])] + [(f"unit {i}", [i]) for i in WAVE] + [
    ("units 2, 10, 11, 12, 13 (small clips)", [2, 10, 11, 12, 13]),
    ("units 2, 10, 11, 12, 13, 14", [2, 10, 11, 12, 13, 14]),
    ("units 3, 15", [3, 15]),
    ("units 0, 3, 15", [0, 3, 15]),
    ("all wave units except 0", [i for i in WAVE if i != 0]),
    ("all 9 wave units", WAVE),
    ("control: comparator 8", [8]), ("control: comparator 7", [7]), ("control: pulse 5", [5]), ("control: pulse 9", [9]), ("control: pulse 4", [4])]
res = {name: np.zeros(3) for name, _ in sets}
res1 = {name: np.zeros(3) for name, _ in sets}
n = 0
for seed, x, H, Z in batches():
    lab = second_argmax(x); ans0 = logits(H).argmax(1); n += len(x)
    for name, lin in sets:
        a = run_lin(x, lin).argmax(1)
        res[name] += [(a == lab).sum(), (a != ans0).sum(), ((a == lab) & (ans0 != lab)).sum()]
        a = run_lin(x, lin, t_from=1).argmax(1)
        res1[name] += [(a == lab).sum(), (a != ans0).sum(), ((a == lab) & (ans0 != lab)).sum()]
rows = []
base = res["none (original)"][0] / n
for name, _ in sets:
    r, r1 = res[name], res1[name]
    rows.append([name, f"{r[0] / n:.4f}", f"{(r[0] / n - base) * 100:+.2f}", f"{r[1] / n:.4f}", f"{r1[0] / n:.4f}", f"{r1[1] / n:.4f}"])
print(md_table(["units made linear (h = z)", "accuracy", "change (pp)", "P(answer differs from original)",
                "accuracy, linear from t=1 only", "P(answer differs), from t=1"], rows))
