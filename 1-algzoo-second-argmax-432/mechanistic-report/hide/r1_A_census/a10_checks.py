"""Independent re-implementation checks (plain Python loops, fresh seed 1100, 30000 sequences).
Recomputes classes, lagged rules and thresholds without cc.events / a02 helpers, to catch indexing mistakes."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
import numpy as np
from common import w, W, U, sample, second_argmax

x = sample(30000, 1100)
B, T = x.shape
cnt = {}


def tally(name, pred, truth):
    c = cnt.setdefault(name, [0, 0])
    c[0] += int(pred == truth); c[1] += 1


ncorrect = 0; mism_thr = 0; n_rest = 0; n_steps = 0
cls_count = {"R": 0, "N": 0, "L": 0}
for b in range(B):
    h = np.zeros(16); gates = []; eff = []
    seen = []
    for t in range(T):
        z = w * x[b, t] + W @ h
        g = z > 0
        # exact thresholds
        for j in (1, 6, 7, 8):
            mism_thr += int((x[b, t] < -(W[j] @ h) / w[j]) != g[j])
        mism_thr += int((x[b, t] > -(W[4] @ h) / w[4]) != g[4])
        srt = sorted(seen, reverse=True)
        M = srt[0] if len(srt) >= 1 else -np.inf
        S = srt[1] if len(srt) >= 2 else -np.inf
        if t >= 1:
            cls_count["R" if x[b, t] > M else ("N" if x[b, t] > S else "L")] += 1
        e = "R" if x[b, t] > max(M, 0) else ("N" if x[b, t] > max(S, 0) else "L")
        eff.append(e); gates.append(g); seen.append(x[b, t])
        h = np.maximum(z, 0)
        n_steps += 1
        n_rest += int(all(g[i] for i in (0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15)) and not (g[4] or g[5] or g[9]))
        tally("4 on(t) <=> R*(t)", e == "R", g[4])
        tally("4 on(t) <=> 7 off(t)", not g[7], g[4])
        if t >= 1:
            tally("9 on(t) <=> 4 on(t-1)", gates[t - 1][4], g[9])
            tally("9 on(t) <=> 4 on(t-1) and 8 off(t-1)", gates[t - 1][4] and not gates[t - 1][8], g[9])
            tally("5 on(t) <=> R*(t-1) or N*(t-1)", eff[t - 1] != "L", g[5])
            tally("5 on(t) <=> x_(t-1) > 0.75 S+ (and > 0)", x[b, t - 1] > 0.75 * max(Sprev_prev, 0) and x[b, t - 1] > 0, g[5])
            tally("0 off(t) <=> 7 on(t-1) and 9 on(t-1)", gates[t - 1][7] and gates[t - 1][9], not g[0])
        if t >= 2:
            tally("0 off(t) <=> R*(t-2) and not R*(t-1)", eff[t - 2] == "R" and eff[t - 1] != "R", not g[0])
            tally("0 off(t) <=> 7 on(t-1) and 9 on(t-1) [t>=2]", gates[t - 1][7] and gates[t - 1][9], not g[0])
        Sprev_prev = S                                   # S of the prefix before step t, used at step t+1 for x_t
    ncorrect += int(np.argmax(U @ h) == np.argsort(x[b])[-2])
print("accuracy:", ncorrect / B)
print("exact threshold mismatches:", mism_thr, "of", 5 * n_steps)
print("resting share:", n_rest / n_steps)
tot = sum(cls_count.values())
print("literal class shares t>=1:", {k: round(v / tot, 4) for k, v in cls_count.items()})
for k, (a, n) in cnt.items():
    print(f"{k}: accuracy {a / n:.4f}  (n = {n})")
