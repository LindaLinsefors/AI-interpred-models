"""v3: functional tests of 'the answer follows the runner-up age stored in the tape', my own code, seed 303.
A. quiet final states: tape set to zero; transplants (rest of recipient + tape of donor scaled by M_rec/M_don);
   tape made k resting steps older (c_m -> lam_m^k c_m), all four modes or some; control: the CLOCK (not tape) made older.
B. ATTACK, other population: the same at EARLIER steps t = 3..8 with the run continued on the real later inputs,
   on the states after which no later input exceeds S_t (so the untouched answer should stay the runner-up of step t).
C. ATTACK, other population: A on uniform inputs on [-1, 1].
Output: v3_functional.txt"""
from vcommon import *

out = Tee(__file__.replace(".py", ".txt"))
neg = lambda hn, M: (hn.min(1) < -0.01 * M).mean()


def donors(rng, aM, aS, same_M, same_S, extra=None):
    """a random other state with equal / different a_M, a_S as asked (and equal `extra` label if given)."""
    B = len(aM); d = rng.integers(0, B, B)
    for _ in range(300):
        bad = ((aM[d] == aM) != same_M) | ((aS[d] == aS) != same_S) | (d == np.arange(B))
        if extra is not None: bad |= extra[d] != extra
        if not bad.any(): break
        d[bad] = rng.integers(0, B, bad.sum())
    return d, ~bad


def part_A(dist, seed, label):
    x, h, st = quiet_final(400_000, seed, dist)
    M, aM, aS, pS, pM = (st[k] for k in ("M", "aM", "aS", "pS", "pM"))
    B = len(x); rng = np.random.default_rng(seed + 1); c = tape_coords(h); a0 = answer(h); bef = aS > aM
    out(f"\n===== {label}: quiet final states {B}; untouched answer = runner-up {np.mean(a0 == pS):.3f}")
    z = answer(set_tape(h, 0 * c))
    out(f"tape = 0: answer = runner-up {np.mean(z == pS):.3f}, = maximum {np.mean(z == pM):.3f}, = last position (9) {np.mean(z == 9):.3f}, same as untouched {np.mean(z == a0):.3f}")
    out("transplants:   answer = tape owner's runner-up | rest owner's runner-up | rest owner's maximum | share with a unit < -0.01 M   (as is / clipped at 0)")
    for nm, sm, ss in (("same a_M, same a_S", True, True), ("same a_M, different a_S", True, False), ("different a_M, same a_S", False, True), ("different a_M and a_S", False, False)):
        d, ok = donors(rng, aM, aS, sm, ss)
        if not (sm or ss): ok &= pS[d] != pM
        hn = set_tape(h, c[d] * (M / M[d])[:, None])
        for clip in (False, True):
            a = answer(np.maximum(hn, 0) if clip else hn)[ok]
            out(f"   {nm:26s} {'clipped' if clip else 'as is':7s} n {ok.sum():6d} | {np.mean(a == pS[d][ok]):.3f} | {np.mean(a == pS[ok]):.3f} | {np.mean(a == pM[ok]):.3f} | {neg(hn[ok], M[ok]):.3f}")
        if sm and not ss:
            a = answer(hn)
            out("      split, answer = tape owner's runner-up:  recipient S before M %.3f, after %.3f | donor S before M %.3f, after %.3f | donor a_S = 2..9: %s" % (
                np.mean((a == pS[d])[bef]), np.mean((a == pS[d])[~bef]), np.mean((a == pS[d])[bef[d]]), np.mean((a == pS[d])[~bef[d]]),
                " ".join(f"{np.mean((a == pS[d])[aS[d] == k]):.2f}" for k in range(2, 10))))
    out("tape made k steps older:   answer = p_S - k | still p_S | unit < -0.01 M     (as is / clipped)")
    for k in (1, 2, 3, -1, -2):
        hn = set_tape(h, advance(c, k)); tgt = pS - k; valid = (aS + k >= 2) & (aS + k <= 9)
        for nm, keep in (("target not the maximum", valid & (tgt != pM)), ("target is the maximum", valid & (tgt == pM))):
            a, ac = answer(hn)[keep], answer(np.maximum(hn, 0))[keep]
            out(f"   k = {k:+d}, {nm:24s} n {keep.sum():6d} | {np.mean(a == tgt[keep]):.3f} / {np.mean(ac == tgt[keep]):.3f} | {np.mean(a == pS[keep]):.3f} / {np.mean(ac == pS[keep]):.3f} | {neg(hn[keep], M[keep]):.3f}")
    keep = (aS + 1 <= 9) & (pS - 1 != pM); tgt = pS - 1
    out(f"only some coordinates one step older (n {keep.sum()}):   answer = p_S - 1 / still p_S   [all | S before M | S after M]")
    for sub in (("short",), ("h2",), ("h3",), ("alt",), ("h2", "h3", "alt"), ("short", "h2", "h3"), TAPE):
        a = answer(set_tape(h, advance(c, 1, sub)))
        out(f"   {' + '.join(sub):22s} " + " | ".join(f"{np.mean((a == tgt)[k]):.3f} / {np.mean((a == pS)[k]):.3f}" for k in (keep, keep & bef, keep & ~bef)))
    hc = h.copy(); zc = mode(h, "clock"); hc[:, ORD] += 2 * np.real(np.multiply.outer(zc * (LAM["clock"] - 1), RV["clock"]))
    a = answer(hc)
    out(f"   CONTROL clock (not tape) one step older: answer = p_S - 1 {np.mean((a == tgt)[keep]):.3f} / still p_S {np.mean((a == pS)[keep]):.3f} / maximum {np.mean((a == pM)[keep]):.3f}")


part_A("normal", 303, "N(0,1) inputs")

out("\n===== B. earlier steps t = 3..8, run continued (N(0,1), 300,000 sequences, seed 304)")
x, H, seq, step, st, hq = pooled(300_000, 304)
M, S, aM, aS, pS, pM = (st[k] for k in ("M", "S", "aM", "aS", "pS", "pM"))
base = answer(H[:, 9]); c = tape_coords(hq); rng = np.random.default_rng(5)
later_max = np.where(np.arange(10)[None] > step[:, None], x[seq], -np.inf).max(1)
calm = later_max < S                                           # no later input above the runner-up of step t
out("states at step t after which no later input exceeds S_t ('calm'); final answer of the continued run")
out("   t      n  | untouched = p_S | tape one step older: = p_S - 1 / still p_S (clipped: = p_S - 1) | transplant, same a_M, other a_S, same step: tape owner's / rest owner's runner-up position (clipped) | tape = 0: = p_S")
for t in range(3, 9):
    k = np.flatnonzero((step == t) & calm); fut = x[seq[k], t + 1:]
    fin = lambda hn: answer(run(fut, h0=hn)[:, -1])
    u = base[seq[k]] == pS[k]
    ok = (aS[k] + 1 <= t) & (pS[k] - 1 != pM[k])
    hn = set_tape(hq[k], advance(c[k], 1)); a, ac = fin(hn), fin(np.maximum(hn, 0))
    d, okd = donors(rng, aM[k], aS[k], True, False)
    okd &= later_max[k] < S[k][d] * M[k] / M[k][d]             # ... nor the donor's runner-up in the recipient's units
    hd = set_tape(hq[k], c[k][d] * (M[k] / M[k][d])[:, None]); ad, adc = fin(hd), fin(np.maximum(hd, 0))
    z = fin(set_tape(hq[k], 0 * c[k]))
    out(f"   {t}  {len(k):6d} | {u.mean():.3f} | {np.mean((a == pS[k] - 1)[ok]):.3f} / {np.mean((a == pS[k])[ok]):.3f} ({np.mean((ac == pS[k] - 1)[ok]):.3f}) | "
        f"{np.mean((ad == pS[k][d])[okd]):.3f} / {np.mean((ad == pS[k])[okd]):.3f} ({np.mean((adc == pS[k][d])[okd]):.3f}) | {np.mean(z == pS[k]):.3f}")

part_A("uniform", 305, "C. uniform inputs on [-1, 1]")
