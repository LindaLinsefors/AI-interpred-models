"""v4: the tape rebuilt from named quantities, my own code, seed 404.  300,000 N(0,1) sequences, quiet states of t = 3..9
pooled; tables fitted on the states of the first 225,000 sequences, everything scored on the other 75,000.
    c / M = A_g + B_g v      E_S: g = a_S | E1: g = (a_M, a_S) | E3: g = (a_M, a_S), v = S/M
    E4: g = (a_M, a_S, a_T), v = (S/M, T+/M)  (classes with < 40 training states: E3) | ADD: A1(a_M) + A2(a_S) (+ the same times S/M)
The tape is replaced at step t and the run continued on the real later inputs.  Reported per step:
    acc     final answer = true second argmax          (the investigator's score)
    agree   final answer = the untouched run's answer   (what 'does not change what the network answers' means)
    broken  P(wrong | untouched right)      fixed  P(right | untouched wrong)      same error  P(same answer | untouched wrong)
ATTACK (oracle control): the same table E3 read at a FALSE runner-up age (the third's age a_T instead of a_S).
If the network then answers the third, 'acc' mainly measures the answer that was written into the rebuilt tape.
Output: v4_rebuilt.txt"""
from vcommon import *

out = Tee(__file__.replace(".py", ".txt"))
n, ntr = 300_000, 225_000
x, H, seq, step, st, hq = pooled(n, 404)
M, S, T3, aM, aS, aT, pS, pT = (st[k] for k in ("M", "S", "T", "aM", "aS", "aT", "pS", "pT"))
truth, base = second_argmax(x), answer(H[:, 9])
c = tape_coords(hq) / M[:, None]; tr = seq < ntr; te = ~tr; one = np.ones(len(seq)); r = S / M; Tp = np.maximum(T3, 0) / M
out(f"{n} sequences (seed 404); quiet states t = 3..9: {len(seq)}, train {tr.sum()}, test {te.sum()}")


def table(group, V):
    """per-class least squares of c/M on [1, V] (training states); returns {class: coefficient matrix}."""
    X = np.column_stack([one] + V); tab = {}
    for g in np.unique(group[tr]):
        i = np.flatnonzero(tr & (group == g))
        if len(i) >= 40: tab[g] = np.linalg.lstsq(X[i], c[i], rcond=None)[0]
    return tab


def read(tab, group, V):
    X = np.column_stack([one] + V); p = np.full(c.shape, np.nan)
    for g in np.unique(group):
        if g in tab: i = group == g; p[i] = X[i] @ tab[g]
    return p


g1, g2, g3 = aS, aM * 10 + aS, aM * 100 + aS * 10 + aT
t3 = table(g2, [r]); E3 = read(t3, g2, [r])
E4 = read(table(g3, [r, Tp]), g3, [r, Tp]); fb = np.isnan(E4[:, 0]); E4[fb] = E3[fb]
oh = lambda a: (a[:, None] == np.arange(2, 10)).astype(float)
Xadd = np.hstack([oh(aM), oh(aS), oh(aM) * r[:, None], oh(aS) * r[:, None]])
ADD = Xadd @ np.linalg.lstsq(Xadd[tr], c[tr], rcond=None)[0]
models = {"zero": 0 * c, "E_S  a_S": read(table(g1, []), g1, []), "E1   (a_M,a_S)": read(table(g2, []), g2, []), "E3   (a_M,a_S)+S/M": E3,
          "E4   +a_T, T+/M": E4, "ADD  A1(a_M)+A2(a_S)": ADD}
assert not any(np.isnan(p[te]).any() for p in models.values())
var = c[tr].var(0)
ev = lambda p, k: 1 - (((c - p)[k] ** 2).mean(0) / var).mean()
out("variance of c/M explained on test states (7 coordinates, each in units of its training variance):")
out(f"   {'model':22s} {'pooled':>7s}" + "".join(f"{'t=' + str(t):>7s}" for t in range(3, 10)))
for nm, p in models.items():
    out(f"   {nm:22s} {ev(p, te):7.3f}" + "".join(f"{ev(p, te & (step == t)):7.3f}" for t in range(3, 10)))


def final(p, k, t, clip=False):
    hn = set_tape(hq[k], p[k] * M[k, None])
    if clip: hn = np.maximum(hn, 0)
    return answer(hn) if t == 9 else answer(run(x[seq[k], t + 1:], h0=hn)[:, -1])


out("\nper step of the replacement: acc / agree / broken / fixed / same error   (test states)")
out(f"   {'':22s}" + "".join(f"{'t = ' + str(t):>34s}" for t in range(3, 10)))
fin = {}
for nm, p in models.items():
    cells = []
    for t in range(3, 10):
        k = np.flatnonzero(te & (step == t)); a = final(p, k, t); fin[nm, t] = a
        tru, b = truth[seq[k]], base[seq[k]]; ur = b == tru
        cells.append(f"{np.mean(a == tru):.3f} / {np.mean(a == b):.3f} / {np.mean((a != tru)[ur]):.3f} / {np.mean((a == tru)[~ur]):.3f} / {np.mean((a == b)[~ur]):.3f}")
    out(f"   {nm:22s}" + "".join(f"{v:>34s}" for v in cells))
out(f"   {'untouched acc':22s}" + "".join(f"{np.mean(base[seq[k]] == truth[seq[k]]):>34.3f}" for t in range(3, 10) for k in [np.flatnonzero(te & (step == t))]))
out(f"   {'number of test states':22s}" + "".join(f"{int((te & (step == t)).sum()):>34d}" for t in range(3, 10)))
k9 = np.flatnonzero(te & (step == 9))
out("   clipped at zero, acc for t = 3..9:  E3 " + " ".join(f"{np.mean(final(E3, kk, t, True) == truth[seq[kk]]):.3f}" for t in range(3, 10) for kk in [np.flatnonzero(te & (step == t))])
    + f";  share of rebuilt final states with a unit < -0.01 M (E3, t = 9): {np.mean(set_tape(hq[k9], E3[k9] * M[k9, None]).min(1) < -0.01 * M[k9]):.3f}")

rng = np.random.default_rng(9); cells = []
for t in range(3, 10):                                         # real donor: random other state of the same step and (a_M, a_S)
    k = np.flatnonzero(te & (step == t)); o = k[np.lexsort((rng.random(len(k)), g2[k]))]
    same = g2[o][1:] == g2[o][:-1]; rec, don = o[1:][same], o[:-1][same]
    hn = set_tape(hq[rec], c[don] * M[rec, None])
    a = answer(hn) if t == 9 else answer(run(x[seq[rec], t + 1:], h0=hn)[:, -1])
    cells.append(f"{np.mean(a == truth[seq[rec]]):.3f} / {np.mean(a == base[seq[rec]]):.3f}")
out(f"   {'real donor, same (a_M,a_S): acc / agree':42s}" + "  ".join(cells))

out("\nsplit by what happens AFTER step t (test states, t = 3..8 pooled): acc untouched | E1 | E3 | E4 ; agree E3 | E4")
k = np.flatnonzero(te & (step < 9))
later = np.where(np.arange(10)[None] > step[k, None], x[seq[k]], -np.inf)
nA, nB = (later > M[k, None]).sum(1), ((later > S[k, None]) & (later <= M[k, None])).sum(1)
cat = lambda nm: np.concatenate([fin[nm, t] for t in range(3, 9)])
tru, b = truth[seq[k]], base[seq[k]]
for nm, sel in (("no later input above S_t", (nA == 0) & (nB == 0)), ("later input(s) between S_t and M_t only", (nA == 0) & (nB > 0)),
                ("exactly one later input above M_t, none between", (nA == 1) & (nB == 0)), ("other", (nA > 0) & (nA + nB > 1))):
    out(f"   {nm:48s} n {sel.sum():6d} | {np.mean((b == tru)[sel]):.3f} | " + " | ".join(f"{np.mean((cat(m) == tru)[sel]):.3f}" for m in ("E1   (a_M,a_S)", "E3   (a_M,a_S)+S/M", "E4   +a_T, T+/M"))
        + " ; " + " | ".join(f"{np.mean((cat(m) == b)[sel]):.3f}" for m in ("E3   (a_M,a_S)+S/M", "E4   +a_T, T+/M")))

out("\nATTACK, oracle control: table E3 read at the THIRD's age in place of the runner-up's, c/M = A(a_M, a_T) + B(a_M, a_T) S/M")
out("   states with a_T >= 2 (a_T != a_M always); 'calm' = no later input above the third T_t, so nothing later should change the top three")
out("   t       n | final answer = third's position | = runner-up's position | untouched run: = runner-up's position")
gl = aM * 10 + aT; okl = te & (aT >= 2); LIE = read(t3, np.where(okl, gl, -1), [r])
later_all = np.where(np.arange(10)[None] > step[:, None], x[seq], -np.inf).max(1)
for t in range(3, 10):
    k = np.flatnonzero(okl & (step == t) & (later_all < T3) & ~np.isnan(LIE[:, 0]))
    if len(k) == 0: continue
    a = final(LIE, k, t)
    out(f"   {t}  {len(k):6d} | {np.mean(a == pT[k]):.3f} | {np.mean(a == pS[k]):.3f} | {np.mean(base[seq[k]] == pS[k]):.3f}")
