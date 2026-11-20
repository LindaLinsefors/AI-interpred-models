"""V7 attack 2: is the maximum's position, in the scores without comparators m = U[:, non-COMP] h_9[non-COMP],
'the one that is not voted against', and do the silent units 0, 12, 11, 10 cause it?

Convention: 'top of m' = argmax over positions 0..8 (the current input, index 9, is left out, as in the lead's
claim S1); the all-ten-positions number is printed once for comparison.
Tests (all at t = 9; seeds 5021..5025, 400k sequences each; selection M > 0.3 and a_1 >= 1 unless stated):
  A. per age a_1: P(top of m = p_1); the same from the negative votes alone and from the positive votes alone;
     who votes at the maximum's position.
  B. necessity: force unit j to keep its previous-step value (h_9[j] <- h_8[j], the same sequence one step
     earlier, i.e. its value at age a* - 1) and see what m and the full logits then rank first.
     Full unit x age matrix, so every other wave unit and every other age is a control.
  C. dose controls: the same amount, but only on / everywhere except the maximum's position.
  D. sufficiency: silence unit j when the max is NOT at the matching age.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *

SEEDS = [5021, 5022, 5023, 5024, 5025]
NB = 400_000
NC = PULSE + WAVE                       # non-comparator units
PAIRS = [(0, 2), (12, 4), (11, 5), (10, 6)]

Hs, H8s, xs = [], [], []
for s in SEEDS:
    x = sample(NB, s)
    H = run(x)
    Hs.append(H[:, 9]); H8s.append(H[:, 8]); xs.append(x)
h9, h8, x = np.concatenate(Hs), np.concatenate(H8s), np.concatenate(xs)
del Hs, H8s, xs
B = len(x)
order = np.argsort(-x, axis=1, kind="stable")
p1, p2 = order[:, 0], order[:, 1]
Mx, Sx = x[np.arange(B), p1], x[np.arange(B), p2]
a1 = 9 - p1
print(f"seeds {SEEDS}, {NB} each, {B} sequences; t = 9 states")

Unc = U[:, NC]
m = h9[:, NC] @ Unc.T                                  # scores without comparators
V = h9[:, COMP] @ U[:, COMP].T
s = m + V
ans = s.argmax(1)
print(f"network accuracy (answer = p2): {(ans == p2).mean():.4f}")
sel = (Mx > 0.3) & (a1 >= 1)


def top(mm):                                           # top of m among positions 0..8
    return mm[:, :9].argmax(1)


# ------------------------------------------------------------------ A
Qneg = np.empty_like(m); Ppos = np.empty_like(m)       # total negative / positive vote per position
for i in range(0, B, 100_000):
    c = h9[i:i + 100_000][:, None, NC] * Unc[None, :, :]
    Qneg[i:i + 100_000] = np.minimum(c, 0).sum(2)
    Ppos[i:i + 100_000] = np.maximum(c, 0).sum(2)
del c
print("\n===== A. per age of the max (t = 9, M > 0.3): which position do the comparator-free scores m rank first? =====")
print("  P10 = P(argmax over all ten positions = p1);  P9 = P(argmax over positions 0..8 = p1);  P9[S>0]: selection S > 0 instead of M > 0.3")
print("  Pneg = P(position 0..8 with the smallest total negative vote = p1);  Ppos = P(position 0..8 with the largest total positive vote = p1)")
print("  means in units of M: m[p1], best other m (0..8), negative votes at p1 and mean over the other positions 0..8, the same for positive votes")
print("  a1  pos      n     P10      P9   P9[S>0]    Pneg    Ppos  |  m[p1]  best other |  neg[p1]  neg[other] |  pos[p1]  pos[other]")
for a in range(1, 10):
    k = (Mx > 0.3) & (a1 == a)
    kS = (Sx > 0) & (a1 == a)
    n = k.sum(); ar = np.arange(n); pk = p1[k]; Mk = Mx[k]
    mm, qq, pp = m[k], Qneg[k], Ppos[k]
    oth = np.ones((n, 10), bool); oth[ar, pk] = False; oth[:, 9] = False
    print(f"  {a:2d}  {9 - a:3d} {n:7d}  {(mm.argmax(1) == pk).mean():.4f}  {(top(mm) == pk).mean():.4f}  {(top(m[kS]) == p1[kS]).mean():.4f}   "
          f"{(top(qq) == pk).mean():.4f}  {(top(pp) == pk).mean():.4f}  | {(mm[ar, pk] / Mk).mean():6.2f}  {(np.where(oth, mm, -np.inf).max(1) / Mk).mean():6.2f}     |"
          f"  {(qq[ar, pk] / Mk).mean():7.2f}  {((qq * oth).sum(1) / 8 / Mk).mean():7.2f}     | {(pp[ar, pk] / Mk).mean():7.2f}  {((pp * oth).sum(1) / 8 / Mk).mean():7.2f}")
print("  rank of the max's position among positions 0..8 (0 = best) by negative votes alone / by positive votes alone, and the position that is most often least voted against:")
for a in range(1, 10):
    k = (Mx > 0.3) & (a1 == a)
    n = k.sum(); ar = np.arange(n); pk = p1[k]
    qq, pp = Qneg[k][:, :9], Ppos[k][:, :9]
    rq = (qq > qq[ar, pk][:, None]).sum(1); rp = (pp > pp[ar, pk][:, None]).sum(1)
    lead = np.bincount(qq.argmax(1), minlength=9)
    print(f"    a1={a} (pos {9 - a}): mean rank by negative votes {rq.mean():.2f}, by positive votes {rp.mean():.2f};  least voted against: position {lead.argmax()} in {lead.max() / n:.3f} of sequences")
kk = (Sx > 0) & (p1 < 9)
print(f"  [lead's S1 at t=9 (S>0, p1<9, positions 0..8): P(argmax m = p1) = {(top(m[kk]) == p1[kk]).mean():.4f}]")

print("\n  who votes at the maximum's position: mean of U[p1, j] h_9[j] / M per non-comparator unit (columns) and age (rows)")
print("  a1  pos " + "".join(f"   u{j:<4d}" for j in NC) + "    sum")
for a in range(1, 10):
    k = sel & (a1 == a)
    v = (h9[k][:, NC] * Unc[9 - a][None, :] / Mx[k, None]).mean(0)
    print(f"  {a:2d}  {9 - a:3d} " + "".join(f"{q:8.1f}" for q in v) + f"  {v.sum():7.1f}")
print("  for comparison: mean vote of each unit on a position that is NOT the max's (average over the other positions 0..8)")
for a in range(1, 10):
    k = sel & (a1 == a)
    others = [p for p in range(9) if p != 9 - a]
    v = (h9[k][:, NC] / Mx[k, None]).mean(0) * Unc[others].mean(0)      # linear, so the two means factorise
    print(f"  {a:2d}  oth " + "".join(f"{q:8.1f}" for q in v) + f"  {v.sum():7.1f}")

print("\n  size of the silent units at the matching age vs elsewhere (h_j / M, t = 9, M > 0.3):")
for j, a in PAIRS:
    k = sel & (a1 == a); r = h9[k, j] / Mx[k]
    ko = sel & (a1 != a); ro = h9[ko, j] / Mx[ko]
    print(f"    unit {j:2d} at a1={a}: P(=0) {np.mean(r == 0):.3f}  mean {r.mean():6.3f}  median {np.median(r):6.3f}  p90 {np.quantile(r, .9):6.3f}  p99 {np.quantile(r, .99):6.3f}"
          f"   | other ages: mean {ro.mean():6.2f}  median {np.median(ro):6.2f};   its vote on position {9 - a}, U[{9 - a},{j}]*h/M: mean {U[9 - a, j] * r.mean():7.2f} (other ages: {U[9 - a, j] * ro.mean():7.2f})")


# ------------------------------------------------------------------ B
def report(label, k, m2):
    s2 = m2 + V[k]
    return (f"{label:46s} P(top of m = p1): {(top(m[k]) == p1[k]).mean():.3f} -> {(top(m2) == p1[k]).mean():.3f};  "
            f"P(answer = p2): {(ans[k] == p2[k]).mean():.3f} -> {(s2.argmax(1) == p2[k]).mean():.3f};  "
            f"P(answer = p1): {(ans[k] == p1[k]).mean():.3f} -> {(s2.argmax(1) == p1[k]).mean():.3f}")


def margin(mm, pk, Mk):
    ar = np.arange(len(pk))
    oth = np.ones(mm[:, :9].shape, bool); oth[ar, pk] = False
    return np.median((mm[ar, pk] - np.where(oth, mm[:, :9], -np.inf).max(1)) / Mk)


print("\n===== B. necessity: h_9[j] <- h_8[j] (unit j keeps the value it had one step earlier), t = 9, M > 0.3 =====")
for j, a in PAIRS:
    k = sel & (a1 == a)
    d = h8[k, j] - h9[k, j]
    m2 = m[k] + d[:, None] * U[None, :, j]
    print("  " + report(f"unit {j} at a1={a} (max at position {9 - a})", k, m2))
    tp = top(m2)
    print(f"      forced value h_8[{j}]/M: median {np.median(h8[k, j] / Mx[k]):.2f};  new top of m: " +
          ", ".join(f"pos {q}: {np.mean(tp == q):.3f}" for q in np.argsort(-np.bincount(tp, minlength=10))[:4]) +
          f";  P(new top = p2): {(tp == p2[k]).mean():.3f} (before: {(top(m[k]) == p2[k]).mean():.3f})")
    print(f"      margin (m[p1] - best other position 0..8)/M, median: {margin(m[k], p1[k], Mx[k]):.2f} -> {margin(m2, p1[k], Mx[k]):.2f};  "
          f"push on p1: median {np.median(d * U[9 - a, j] / Mx[k]):.2f} M")

print("\n  full matrix: P(top of m' = p1) after h_9[k] <- h_8[k]; rows = age of the max, columns = unit k forced")
print("  a1   none  " + "".join(f"   u{k:<3d}" for k in WAVE))
for a in range(1, 10):
    km = sel & (a1 == a)
    row = f"  {a:2d}  {(top(m[km]) == p1[km]).mean():.3f} "
    for k in WAVE:
        row += f"  {(top(m[km] + (h8[km, k] - h9[km, k])[:, None] * U[None, :, k]) == p1[km]).mean():.3f}"
    print(row)
print("\n  the same for the network's answer: P(answer' = p2) (full logits, comparators included)")
print("  a1   none  " + "".join(f"   u{k:<3d}" for k in WAVE))
for a in range(1, 10):
    km = sel & (a1 == a)
    row = f"  {a:2d}  {(ans[km] == p2[km]).mean():.3f} "
    for k in WAVE:
        row += f"  {((s[km] + (h8[km, k] - h9[km, k])[:, None] * U[None, :, k]).argmax(1) == p2[km]).mean():.3f}"
    print(row)

# ------------------------------------------------------------------ C
print("\n===== C. dose controls for B (same sequences, same amount h_8[j] - h_9[j]) =====")
for j, a in PAIRS:
    k = sel & (a1 == a)
    d = h8[k, j] - h9[k, j]
    pm = 9 - a
    m2 = m[k].copy(); m2[:, pm] += d * U[pm, j]
    print("  " + report(f"unit {j}, a1={a}: only its vote on position {pm}", k, m2))
    col = U[:, j].copy(); col[pm] = 0
    print("  " + report(f"unit {j}, a1={a}: all its votes except on {pm}", k, m[k] + d[:, None] * col[None, :]))

# ------------------------------------------------------------------ D
print("\n===== D. sufficiency: silence unit j (h_9[j] <- 0) when the max is NOT at the matching age =====")
print("  q = the unit's matching position. Entries: P(top of m = q) before -> after | P(top of m = p1) before -> after")
for j, a in PAIRS:
    q = 9 - a
    print(f"  unit {j:2d} (q = {q}, U[q, j] = {U[q, j]:.2f}):")
    for b in range(1, 10):
        if b == a:
            continue
        k = sel & (a1 == b)
        m2 = m[k] - h9[k, j][:, None] * U[None, :, j]
        print(f"      a1={b} (p1={9 - b}): top=q {np.mean(top(m[k]) == q):.3f} -> {np.mean(top(m2) == q):.3f} | top=p1 {np.mean(top(m[k]) == p1[k]):.3f} -> {np.mean(top(m2) == p1[k]):.3f}"
              f" | removed vote on q: {np.mean(-h9[k, j] * U[q, j] / Mx[k]):6.1f} M")
