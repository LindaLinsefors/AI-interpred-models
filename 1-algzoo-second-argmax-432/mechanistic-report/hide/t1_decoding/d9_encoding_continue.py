"""d9: the functional test of d8 at EARLIER steps, with the run continued.
At step t (t = 3..8), in every quiet state, replace the tape by a tape rebuilt from named quantities only, then feed the
real later inputs x_(t+1)..x_9 and compare the final answer with the true second argmax and with the untouched run.
Models (fitted on quiet states of all steps t = 3..9 pooled, first 75% of the sequences; tested on the other 25%):
    c / M_t = A_g + B_g . v      (c = the 7 tape coordinates, M_t = maximum so far)
    zero   c = 0                          E_S  g = a_S              E_M  g = a_M
    E1     g = (a_M, a_S)                 E3   g = (a_M, a_S), v = (S/M)
    E4     g = (a_M, a_S, a_T), v = (S/M, T+/M)     (classes with < 40 training states fall back to E3)
Also 'donor': the tape of a random other quiet state of the same step with the same (a_M, a_S), scaled by the ratio of M.
Output: d9_encoding_continue.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
n = 300_000; ntr = 225_000
x = sample(n, 21); H = run(x); truth = second_argmax(x); base = answer(H[:, 9])
rng = np.random.default_rng(3)
P = {t: prefix_stats(x, t) for t in range(3, 10)}
rows = [(np.flatnonzero(P[t]["quiet"]), t) for t in range(3, 10)]
seq = np.concatenate([r for r, _ in rows]); step = np.concatenate([np.full(len(r), t) for r, t in rows])
get = lambda k: np.concatenate([P[t][k][r] for r, t in rows])
M, S, T3, aM, aS, aT = (get(k) for k in ("M", "S", "T3", "aM", "aS", "aT"))
hq = H[seq, step]; c = tape_coords(hq) / M[:, None]; tr = seq < ntr; one = np.ones(len(seq))
g2 = aM * 10 + aS; g3 = g2 * 10 + aT
out(f"{n} sequences, quiet states at t = 3..9: {len(seq)}; models fitted on the {tr.sum()} states of the first {ntr} sequences")


def fit(group, V):
    X = np.column_stack([one] + V); pred = np.full(c.shape, np.nan)
    ug, inv = np.unique(group, return_inverse=True)
    o = np.argsort(inv, kind="stable"); cut = np.searchsorted(inv[o], np.arange(len(ug) + 1))
    for k in range(len(ug)):
        idx = o[cut[k]:cut[k + 1]]; t = idx[tr[idx]]
        if len(t) >= 40:
            pred[idx] = X[idx] @ np.linalg.lstsq(X[t], c[t], rcond=None)[0]
    return pred


E3 = fit(g2, [S / M]); E4 = fit(g3, [S / M, np.maximum(T3, 0) / M]); fb = np.isnan(E4[:, 0]); E4[fb] = E3[fb]
models = {"zero": np.zeros_like(c), "E_S  a_S": fit(aS, []), "E_M  a_M": fit(aM, []), "E1   (a_M,a_S)": fit(g2, []),
          "E3   (a_M,a_S) + S/M": E3, "E4   (a_M,a_S,a_T) + S/M, T+/M": E4}
var = c[tr].var(0)
out("share of the variance of c/M explained on the test states: " + "; ".join(f"{k.split()[0]} {1 - (((c - p)[~tr] ** 2).mean(0) / var).mean():.3f}" for k, p in models.items()))
out("")
out("final answer = true second argmax (and, in brackets, = the untouched run's final answer), test sequences, by the step t of the replacement")
out(f"{'tape at step t replaced by':34s}" + "".join(f"{'t = ' + str(t):>16s}" for t in range(3, 10)))
te = ~tr
cnt = [int((te & (step == t)).sum()) for t in range(3, 10)]
out(f"{'number of test states':34s}" + "".join(f"{v:16d}" for v in cnt))
out(f"{'untouched':34s}" + "".join(f"{(base == truth)[seq[te & (step == t)]].mean():16.3f}" for t in range(3, 10)))


def final(hn, k, t, clip=False):
    hn = np.maximum(hn, 0) if clip else hn
    return answer(hn) if t == 9 else answer(continue_from(hn, x[seq[k], t + 1:])[:, -1])


for name, p in models.items():
    for clip in (False, True):
        cells = []
        for t in range(3, 10):
            k = te & (step == t)
            a = final(set_tape(hq[k], p[k] * M[k, None]), k, t, clip)
            cells.append(f"{(a == truth[seq[k]]).mean():.3f} ({(a == base[seq[k]]).mean():.3f})")
        out(f"{name + (', clipped' if clip else ''):34s}" + "".join(f"{v:>16s}" for v in cells))
cells, negs = [], []
for t in range(3, 10):                                       # donor of the same step and the same (a_M, a_S)
    k = np.flatnonzero(te & (step == t)); o = k[np.lexsort((rng.random(len(k)), g2[k]))]
    same = g2[o][1:] == g2[o][:-1]; rec, don = o[1:][same], o[:-1][same]
    hn = rest_part(hq[rec]) + tape_part(hq[don]) * (M[rec] / M[don])[:, None]
    a = answer(hn) if t == 9 else answer(continue_from(hn, x[seq[rec], t + 1:])[:, -1])
    cells.append(f"{(a == truth[seq[rec]]).mean():.3f} ({(a == base[seq[rec]]).mean():.3f})")
out(f"{'donor with the same (a_M,a_S)':34s}" + "".join(f"{v:>16s}" for v in cells))

out("")
out("E1 at step t, split by what happens AFTER t (test states, all t = 3..8 pooled): final answer = true second argmax, modelled / untouched")
k = te & (step < 9)
fin = np.empty(k.sum(), int); pos = 0
for t in range(3, 9):
    kk = te & (step == t); m = models["E1   (a_M,a_S)"]
    fin[pos:pos + kk.sum()] = final(set_tape(hq[kk], m[kk] * M[kk, None]), kk, t); pos += kk.sum()
sq, stp = seq[k], step[k]                                     # k is ordered by step, as the loop above
later = np.where(np.arange(10)[None] > stp[:, None], x[sq], -np.inf)
n_above_M = (later > M[k, None]).sum(1); n_between = ((later > S[k, None]) & (later <= M[k, None])).sum(1)
for nm, sel in (("no later input above S_t", (n_above_M == 0) & (n_between == 0)), ("later input(s) between S_t and M_t only", (n_above_M == 0) & (n_between > 0)),
                ("exactly one later input above M_t, none between", (n_above_M == 1) & (n_between == 0)), ("everything else", (n_above_M + n_between > 1) & ~((n_above_M == 0)))):
    out(f"   {nm:50s} n {sel.sum():6d}   {(fin == truth[sq])[sel].mean():.3f} / {(base == truth)[sq][sel].mean():.3f}")
