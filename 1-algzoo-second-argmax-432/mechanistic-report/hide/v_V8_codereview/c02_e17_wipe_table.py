"""Code review of e17_wipe_comparators.py (claim I1).

e17 pools over all wipe steps t0 = pM + k <= 9 (its printed header says pM <= 8 - k, the code uses
t0 <= 9), and splits by t0 only for k = 2.  Here: the full (k, t0) table, on the same sample
(seed 1717) and on a fresh one, with the answer classified as max / runner-up / last position 9 /
the wipe step t0 / other.  Also the pooled number without t0 = 9 (the population the header states).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *


def run_wipe(x, t0, units):
    h = np.zeros((len(x), N))
    for t in range(T):
        h = np.maximum(x[:, t, None] * w + h @ W.T, 0)
        if units is not None:
            h[np.ix_(np.where(t0 == t)[0], units)] = 0.0
    return (h @ U.T).argmax(1)


for seed, n in ((1717, 300_000), (91717, 1_000_000)):
    x = sample(n, seed)
    st = running_stats(x)
    truth = second_argmax(x)
    pM = x.argmax(1)
    cond = (st["M2"][:, 9] > 0.2) & (st["M1"][:, 9] - st["M2"][:, 9] > 0.2)
    print(f"\n================ seed {seed}, n = {n}; condition S_9 > 0.2 and M_9 - S_9 > 0.2 ================")
    print("pooled over t0 (as e17) and pooled over t0 <= 8 only (the population e17's header states):")
    print("   k | e17: P(max) P(runner-up) n      | t0<=8 only: P(max) P(runner-up) n      | t0 = 9 only: P(max) P(runner-up) P(answer=9) n")
    tables = {}
    for k in range(1, 9):
        t0 = pM + k
        ok = (t0 <= 9) & cond
        a = run_wipe(x[ok], t0[ok], COMP)
        c = run_wipe(x[ok], t0[ok], [10, 11, 12, 15])
        tm, tr, tt = pM[ok], truth[ok], t0[ok]
        e, l = tt <= 8, tt == 9
        print(f"   {k} |      {np.mean(a == tm):.3f}   {np.mean(a == tr):.3f}    {ok.sum():7d} |"
              f"             {np.mean(a[e] == tm[e]) if e.any() else np.nan:.3f}   {np.mean(a[e] == tr[e]) if e.any() else np.nan:.3f}    {e.sum():7d} |"
              f"              {np.mean(a[l] == tm[l]):.3f}   {np.mean(a[l] == tr[l]):.3f}       {np.mean(a[l] == 9):.3f}     {l.sum():7d}")
        tables[k] = (a, c, tm, tr, tt)
    print("\nP(answer = max) after wiping the comparators, by k (rows) and wipe step t0 (columns):")
    print("   k | " + " ".join(f"t0={t:d}  " for t in range(1, 10)))
    for k, (a, c, tm, tr, tt) in tables.items():
        print(f"   {k} | " + " ".join((f"{np.mean(a[tt == t] == tm[tt == t]):.3f} " if (tt == t).sum() > 200 else "  .   ") for t in range(1, 10)))
    print("\nP(answer = true runner-up) after wiping the comparators:")
    for k, (a, c, tm, tr, tt) in tables.items():
        print(f"   {k} | " + " ".join((f"{np.mean(a[tt == t] == tr[tt == t]):.3f} " if (tt == t).sum() > 200 else "  .   ") for t in range(1, 10)))
    print("\nP(answer = the wipe step t0 itself):")
    for k, (a, c, tm, tr, tt) in tables.items():
        print(f"   {k} | " + " ".join((f"{np.mean(a[tt == t] == t):.3f} " if (tt == t).sum() > 200 else "  .   ") for t in range(1, 10)))
    print("\nP(answer = something else: not max, not runner-up, not t0):")
    for k, (a, c, tm, tr, tt) in tables.items():
        print(f"   {k} | " + " ".join((f"{np.mean((a != tm) & (a != tr) & (a != tt))[()]:.3f} " if False else
                                        f"{np.mean(((a != tm) & (a != tr) & (a != tt))[tt == t]):.3f} ") if (tt == t).sum() > 200 else "  .   " for t in range(1, 10)))
    print("\nn per cell:")
    for k, (a, c, tm, tr, tt) in tables.items():
        print(f"   {k} | " + " ".join(f"{(tt == t).sum():6d}" for t in range(1, 10)))
    print("\ncontrol (wave units 10, 11, 12, 15 wiped): P(answer = max) / P(answer = runner-up), pooled as e17 and for t0 <= 8:")
    for k, (a, c, tm, tr, tt) in tables.items():
        e = tt <= 8
        print(f"   {k} | all t0: {np.mean(c == tm):.3f} / {np.mean(c == tr):.3f}    t0 <= 8: "
              + (f"{np.mean(c[e] == tm[e]):.3f} / {np.mean(c[e] == tr[e]):.3f}" if e.any() else "."))
    # the report's sentence: 'if the wipe is at step 8 or earlier the network answers the max in 99.7-100%'
    a_all = np.concatenate([np.stack([a, tm, tr, tt, np.full(len(a), k)], 1) for k, (a, c, tm, tr, tt) in tables.items()])
    e = a_all[:, 3] <= 8
    print(f"\nall k = 1..8 pooled, wipe at t0 <= 8: P(answer = max) = {np.mean(a_all[e, 0] == a_all[e, 1]):.4f}, "
          f"P(runner-up) = {np.mean(a_all[e, 0] == a_all[e, 2]):.4f}, n = {e.sum()}")
    for k in range(1, 9):
        q = e & (a_all[:, 4] == k)
        if q.any():
            print(f"     k = {k}: P(max) = {np.mean(a_all[q, 0] == a_all[q, 1]):.4f}   P(runner-up) = {np.mean(a_all[q, 0] == a_all[q, 2]):.4f}   n = {q.sum()}")
