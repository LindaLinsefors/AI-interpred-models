"""d6: functional checks of the decoding results.  Quiet final states (t = 9); the answer is argmax U h, so no
continuation is needed.  numpy only.
A. TRANSPLANT.  New state = rest of the recipient + tape part of a donor, the donor's tape scaled by M_rec / M_don.
   Which position does the network answer: the runner-up position of the tape's owner (donor) or of the rest's owner?
   Donor classes: matched on more and more of the decodable quantities, or deliberately different in a_S or a_M.
B. ADVANCE THE TAPE.  Replace the tape coordinates c by Lambda^k c (each coordinate multiplied by its eigenvalue k times,
   i.e. the tape is made k resting steps older; k = -1 makes it one step younger), rest untouched.  Does the answer move
   from the runner-up's position p_S to p_S - k?
C. The same advance by one step applied to SOME of the tape coordinates only, split by configuration.
A and B are repeated with the modified state clipped at zero; the share of states with a negative unit is reported.
Output: d6_functional.txt"""
from dec import *

out = Tee(__file__.replace(".py", ".txt"))
x, h, st, D = data()
B = len(x); rng = np.random.default_rng(5)
M, S, aS, aM, aT, pS, pM = st["M"], st["S"], st["aS"], st["aM"], st["aT"], st["pS"], st["pM"]
Tp = np.maximum(st["T3"], 0)
out(f"quiet final states {B}; untouched, the network answers the runner-up's position in {(D['ans'] == pS).mean():.3f}")


def donors_diff(same_M, same_S):
    """for every state a random other state with equal / different a_M and a_S as asked."""
    d = rng.integers(0, B, B)
    for _ in range(200):
        bad = ((aM[d] == aM) != same_M) | ((aS[d] == aS) != same_S)
        if not bad.any(): break
        d[bad] = rng.integers(0, B, bad.sum())
    return d, ~bad


def donors_group(group, key):
    """the neighbour in the order of `key` within the same group (random key = random member of the group)."""
    o = np.lexsort((key, group)); g = group[o]
    nxt = np.r_[g[1:] == g[:-1], False]; prv = np.r_[False, g[1:] == g[:-1]]
    partner = np.where(nxt, np.roll(o, -1), np.roll(o, 1))
    d = np.empty(B, int); d[o] = partner
    ok = np.empty(B, bool); ok[o] = nxt | prv
    return d, ok


def report(name, d, keep):
    hn = rest_part(h) + tape_part(h[d]) * (M / M[d])[:, None]
    for clip in (False, True):
        a = answer(np.maximum(hn, 0) if clip else hn)[keep]
        tp, rp, mp = pS[d][keep], pS[keep], pM[keep]
        out(f"   {name:50s} {'clipped' if clip else 'as is':7s} n {keep.sum():6d} | answer = tape owner's runner-up {np.mean(a == tp):.3f} | "
            f"rest owner's runner-up {np.mean(a == rp):.3f} | rest owner's maximum {np.mean(a == mp):.3f} | "
            f"other {np.mean((a != tp) & (a != rp) & (a != mp)):.3f} | unit < -0.01 M: {(hn.min(1) < -0.01 * M)[keep].mean():.3f}")


out("")
out("A. transplant of the tape part (donor's tape scaled to the recipient's M)")
r = rng.random(B)
g2 = aM * 10 + aS; g3 = g2 * 10 + aT; b5 = lambda v: np.minimum((5 * v).astype(int), 4)
report("donor: same (a_M, a_S)", *donors_group(g2, r))
report("donor: same (a_M, a_S, a_T)", *donors_group(g3, r))
report("donor: same (a_M, a_S, a_T), nearest (S-T+)/M", *donors_group(g3, (S - Tp) / M))
report("donor: same (a_M,a_S,a_T), S/M in 5 bins, nearest (S-T+)/M", *donors_group(g3 * 10 + b5(S / M), (S - Tp) / M))
report("donor: same (a_M,a_S,a_T), (S-T+)/M in 5 bins, nearest S/M", *donors_group(g3 * 10 + b5((S - Tp) / M), S / M))
report("donor: same a_M, different a_S", *donors_diff(True, False))
report("donor: different a_M, same a_S", *donors_diff(False, True))
d, ok = donors_diff(False, False)
report("donor: different a_M and a_S (*)", d, ok & (pS[d] != pM))
k = ok & (pS[d] == pM)
a = answer(rest_part(h) + tape_part(h[d]) * (M / M[d])[:, None])[k]
out(f"   (*) donors whose runner-up position is the recipient's maximum position are left out of that row. Those {k.sum()} cases: "
    f"answer = that position {np.mean(a == pM[k]):.3f}, recipient's runner-up {np.mean(a == pS[k]):.3f}, other {np.mean((a != pM[k]) & (a != pS[k])):.3f}")

out("")
out("B. the tape made k steps older: c -> Lambda^k c.  'target' = position p_S - k.")
c = tape_coords(h)
cols = {"alt": [0], "short": [1, 2], "h2": [3, 4], "h3": [5, 6]}


def advance(c, k):
    o = np.empty_like(c); o[:, 0] = c[:, 0] * LAM["alt"].real ** k
    for j, n in enumerate(("short", "h2", "h3")):
        z = (c[:, 1 + 2 * j] + 1j * c[:, 2 + 2 * j]) * LAM[n] ** k
        o[:, 1 + 2 * j], o[:, 2 + 2 * j] = z.real, z.imag
    return o


for k in (1, 2, 3, -1, -2):
    hn = set_tape(h, advance(c, k)); tgt = pS - k
    valid = (aS + k >= 2) & (aS + k <= 9)
    for nm, keep in ((f"k = {k:+d}, target is not the maximum", valid & (tgt != pM)), (f"k = {k:+d}, target is the maximum", valid & (tgt == pM))):
        for clip in (False, True):
            a = answer(np.maximum(hn, 0) if clip else hn)[keep]
            out(f"   {nm:34s} {'clipped' if clip else 'as is':7s} n {keep.sum():6d} | answer = target {np.mean(a == tgt[keep]):.3f} | "
                f"still p_S {np.mean(a == pS[keep]):.3f} | maximum {np.mean((a == pM[keep]) & (tgt[keep] != pM[keep])):.3f} | "
                f"unit < -0.01 M: {(hn.min(1) < -0.01 * M)[keep].mean():.3f}")

out("")
out("C. only some tape coordinates made one step older (k = +1); states whose target position p_S - 1 exists and is not the maximum")
out("   columns: answer = target / still p_S / maximum / other")
bef = D["S_before_M"]; tgt = pS - 1; ok = (aS + 1 <= 9) & (tgt != pM)
sel = {"all": ok, "S before M": ok & bef, "S after M": ok & ~bef}
out(f"   {'coordinates advanced':24s}" + "".join(f"{k + ' (n ' + str(v.sum()) + ')':>34s}" for k, v in sel.items()))
for sub in (("short",), ("h2",), ("h3",), ("alt",), ("h2", "h3"), ("h3", "alt"), ("h2", "h3", "alt"), ("short", "h2", "h3"),
            ("short", "h3", "alt"), ("short", "h2", "alt"), ("alt", "short", "h2", "h3")):
    cc = sum((cols[m] for m in sub), [])
    cn = c.copy(); cn[:, cc] = advance(c, 1)[:, cc]
    a = answer(set_tape(h, cn))
    row = ""
    for keep in sel.values():
        t, s_, m_ = (a == tgt)[keep].mean(), (a == pS)[keep].mean(), (a == pM)[keep].mean()
        row += f"      {t:.3f} / {s_:.3f} / {m_:.3f} / {1 - t - s_ - m_:.3f}"
    out(f"   {' + '.join(sub):24s}" + row)
