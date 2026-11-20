"""V7 extra causal test in the running network: a record arrives at step t_r < 9; the old maximum (age a at
that step) must end up as the answer.  At step t_r one wave unit is forced to keep the value it had one step
earlier (h_{t_r}[k] <- h_{t_r - 1}[k]); the run then continues with the real inputs.

Prediction of the lead's reading ('travelling zero' + the tape that carries scores forward): forcing unit
0 / 12 / 11 / 10 when the old max is 2 / 4 / 5 / 6 steps old turns the answer away from the old max.
Selection: the largest input is x_{t_r}; the second largest of the whole sequence is the largest of
x_0..x_{t_r - 1} and is > 0.3 (so the true answer is the old max).  Seeds 5051..5055, 400k each.
"""
import sys
from collections import defaultdict
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

SEEDS = [5051, 5052, 5053, 5054, 5055]
NB = 400_000
PAIRS = [(0, 2), (12, 4), (11, 5), (10, 6)]

cnt = defaultdict(int)          # (kind, age, t_r, unit) -> number answering the old max ; ("n", age, t_r) -> count
for seed in SEEDS:
    x = sample(NB, seed)
    H = run(x)
    ans = logits(H).argmax(1)
    order = np.argsort(-x, axis=1, kind="stable")
    p1, p2 = order[:, 0], order[:, 1]
    Sx = x[np.arange(NB), p2]
    ok = (p2 < p1) & (Sx > 0.3)             # old max before the record, old max > 0.3
    age = p1 - p2                           # age of the old max at the record step
    for a in range(1, 10):
        for tr in range(max(a, 2), 10):
            mk = ok & (p1 == tr) & (age == a)
            n = int(mk.sum())
            if n == 0:
                continue
            cnt[("n", a, tr)] += n
            cnt[("base", a, tr)] += int((ans[mk] == p2[mk]).sum())
            for k in WAVE:                  # at the record step
                h = H[mk, tr].copy(); h[:, k] = H[mk, tr - 1, k]
                hf = run(x[mk, tr + 1:], h0=h)[:, -1] if tr < 9 else h
                cnt[("at", a, tr, k)] += int(((hf @ U.T).argmax(1) == p2[mk]).sum())
            if tr >= 3 and tr - 2 >= 0:     # wrong timing: one step before the record
                for k in [0, 12, 11, 10]:
                    h = H[mk, tr - 1].copy(); h[:, k] = H[mk, tr - 2, k]
                    a1 = logits(run(x[mk, tr:], h0=h)).argmax(1)
                    cnt[("before", a, tr, k)] += int((a1 == p2[mk]).sum())
                    cnt[("nb", a, tr)] += n if k == 0 else 0

print(f"seeds {SEEDS}, {NB} each; sequences used: {sum(v for k, v in cnt.items() if k[0] == 'n')}")
print("\n===== matched cells, by record step t_r: P(answer = old max) before -> after forcing the unit AT the record step =====")
for j, a in PAIRS:
    out = f"  unit {j:2d}, old max {a} steps old:"
    for tr in range(max(a, 2), 10):
        n = cnt[("n", a, tr)]
        out += f"  t_r={tr}: {cnt[('base', a, tr)] / n:.3f}->{cnt[('at', a, tr, j)] / n:.3f} (n={n})"
    print(out)

print("\n===== full matrix, records at t_r = 2..8 pooled: P(answer = old max) after forcing unit k at the record step =====")
print("  rows = age of the old max at the record step; columns = unit forced")
print("  age      n    none  " + "".join(f"   u{k:<3d}" for k in WAVE))
for a in range(1, 9):
    trs = range(max(a, 2), 9)
    n = sum(cnt[("n", a, tr)] for tr in trs)
    row = f"  {a:3d} {n:7d}   {sum(cnt[('base', a, tr)] for tr in trs) / n:.3f} "
    for k in WAVE:
        row += f"  {sum(cnt[('at', a, tr, k)] for tr in trs) / n:.3f}"
    print(row)

print("\n===== wrong-timing control: the same forcing one step BEFORE the record (h_{t_r-1}[k] <- h_{t_r-2}[k]), t_r = 3..8 pooled =====")
for j, a in PAIRS:
    trs = range(max(a, 3), 9)
    n = sum(cnt[("nb", a, tr)] for tr in trs)
    b0 = sum(cnt[("base", a, tr)] for tr in trs) / sum(cnt[("n", a, tr)] for tr in trs)
    print(f"  unit {j:2d}, old max {a} steps old at the record: {b0:.3f} -> {sum(cnt[('before', a, tr, j)] for tr in trs) / n:.3f} (n={n});"
          f"   at the record step (same sequences): -> {sum(cnt[('at', a, tr, j)] for tr in trs) / sum(cnt[('n', a, tr)] for tr in trs):.3f}")
