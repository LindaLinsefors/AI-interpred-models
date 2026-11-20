"""Follow-up to c02 (e17): the missing cell k = 9 (max at position 0, wipe at the last step), the
table under the weaker condition used elsewhere (no gap condition), and the comparison with the
passive statistic of S1 (does argmax m_t0 over the older inputs equal the max, by age of the max).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *


def run_wipe(x, t0, units):
    h = np.zeros((len(x), N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        h[np.ix_(np.where(t0 == t)[0], units)] = 0.0
    return (h @ U.T).argmax(1)


x = sample(1_000_000, 91717)
st = running_stats(x)
truth = second_argmax(x)
pM = x.argmax(1)
cond = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2)
print("P(answer = max) after wiping the comparators at t0 = pM + k; rows k = 1..9, columns t0 = 1..9; condition S_9 > 0.2, M_9 - S_9 > 0.2")
allrows = []
for k in range(1, 10):
    t0 = pM + k
    ok = (t0 <= 9) & cond
    a = run_wipe(x[ok], t0[ok], COMP)
    tm, tr, tt = pM[ok], truth[ok], t0[ok]
    print(f"   {k} | " + " ".join((f"{np.mean(a[tt == t] == tm[tt == t]):.3f} " if (tt == t).sum() > 200 else "  .   ") for t in range(1, 10)))
    allrows.append(np.stack([a == tm, a == tr, a == tt, tt, np.full(len(a), k)], 1))
R = np.concatenate(allrows)
for lab, q in (("all k = 1..9, t0 <= 9", np.ones(len(R), bool)), ("t0 <= 8", R[:, 3] <= 8), ("t0 <= 7", R[:, 3] <= 7), ("t0 = 8", R[:, 3] == 8), ("t0 = 9", R[:, 3] == 9)):
    print(f"   pooled, {lab}: P(max) = {R[q, 0].mean():.4f}  P(runner-up) = {R[q, 1].mean():.4f}  P(the wipe step itself) = {R[q, 2].mean():.4f}  n = {q.sum()}")

print("\npassive counterpart (no intervention): P(argmax of m_t over the inputs older than t = position of the max) by age of the max a (rows) and step t (columns 1..9);"
      " same condition on the final sequence, and the max of the prefix is the final max")
H = run(x)
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
Mall = H @ (U - Uc).T
for k in range(1, 10):
    row = []
    for t in range(1, 10):
        q = cond & (pM + k == t)
        if q.sum() < 200:
            row.append("  .   "); continue
        valid = (np.arange(10) >= 9 - t) & (np.arange(10) <= 8)
        pm = np.where(valid, Mall[q, t], -np.inf).argmax(1) + t - 9
        row.append(f"{np.mean(pm == pM[q]):.3f} ")
    print(f"   {k} | " + " ".join(row))
print("   and with index 9 (the current input) allowed, i.e. the literal readout without the comparators at step t:")
for k in range(1, 10):
    row = []
    for t in range(1, 10):
        q = cond & (pM + k == t)
        if q.sum() < 200:
            row.append("  .   "); continue
        valid = (np.arange(10) >= 9 - t)
        pm = np.where(valid, Mall[q, t], -np.inf).argmax(1) + t - 9
        row.append(f"{np.mean(pm == pM[q]):.3f} ")
    print(f"   {k} | " + " ".join(row))
