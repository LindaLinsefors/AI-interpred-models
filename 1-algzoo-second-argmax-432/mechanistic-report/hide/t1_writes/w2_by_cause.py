"""W2. Account for the tape by the INPUT THAT CAUSED each write (quiet final states, t = 9).
Cause of a write at step s:  input term and comparator clips -> x_s;  pulse of unit 4 -> x_(s-1);  pulses of units 5, 9 -> x_(s-2)
(the pulse h_(s-1)[5] answers x_(s-2));  wave-unit clips -> by context, see CONTEXT below.
Class of the causing input j, seen from the final state:
    M        the maximum                               S>M      the runner-up, arrived after the maximum (as a new runner-up)
    S<M      the runner-up, arrived before the maximum (it was the maximum then)
    exmax    was a new maximum at arrival, now third or lower     exrun   was a new runner-up at arrival, now third or lower
    near     at arrival 0.7 S < x <= S (S = runner-up then)        far     0 < x <= 0.7 S       neg   x <= 0
    spont    wave clips at steps with no event in the last two steps (attributed to no input)
Run:  python -B w2_by_cause.py > w2_by_cause.txt"""
from wcommon import *

B = 300_000
x = sample(B, 202)
H, Z, Amp = amplitudes(x)
st = prefix_stats(x, 9); q = st["quiet"]; M = st["M"]; pM, pS = st["pM"], st["pS"]
h9 = H[:, 9]; c9 = tape_coords(h9); ans0 = answer(h9); corr = second_argmax(x)
kind, Mp, Sp, _ = roles(x)
jj = np.arange(10)[None, :]

CLS = ["M", "S>M", "S<M", "exmax", "exrun", "near", "far", "neg", "spont"]
role = np.select([jj == pM[:, None], (jj == pS[:, None]) & (pS > pM)[:, None], (jj == pS[:, None]) & (pS < pM)[:, None],
                  kind == 1, kind == 2, kind == 3, kind == 4], [0, 1, 2, 3, 4, 5, 6], 7)             # (B, 10)

def shift(a, n, fill):
    out = np.full_like(a, fill); out[:, n:] = a[:, : a.shape[1] - n] if n else a
    return out
k0, k1, k2 = kind, shift(kind, 1, 0), shift(kind, 2, 0)
r0, r1, r2_ = role, shift(role, 1, 7), shift(role, 2, 7)
# CONTEXT of a wave clip at step s (first rule that applies): the cause is
ctx = np.select([k0 == 1, k0 == 2, k1 == 1, k2 == 1, (k2 == 2) | (k2 == 3), (k1 == 2) | (k1 == 3)], [0, 1, 2, 3, 4, 5], 6)
CTX = ["x_s is a new max", "x_s is a new runner-up", "x_(s-1) was a new max", "x_(s-2) was a new max",
       "x_(s-2) was a new runner-up / near", "x_(s-1) was a new runner-up / near", "none of these (spontaneous)"]
wave_role = np.select([ctx <= 1, ctx == 2, ctx == 3, ctx == 4, ctx == 5], [r0, r1, r2_, r2_, r1], 8)

# class of every write (B, 10 steps, 17 channels)
wcls = np.empty((B, 10, NCH), dtype=np.int8)
wcls[:, :, 0] = r0; wcls[:, :, COMP_CH] = r0[:, :, None]
wcls[:, :, 1] = r1; wcls[:, :, 2] = r2_; wcls[:, :, 3] = r2_
wcls[:, :, WAVE_CH] = wave_role[:, :, None]

Aq = Amp[q]; Mq = M[q, None]; cq = c9[q] / Mq; sq = cq @ R_SCORE[:8].T; n = q.sum()
def share(part):
    pc = part - part.mean(0); cc = cq - cq.mean(0); sc = pc @ R_SCORE[:8].T; ss = sq - sq.mean(0)
    return (pc * cc).sum(0) / (cc ** 2).sum(0), (sc * ss).sum() / (ss ** 2).sum(), np.sqrt((sc ** 2).sum() / (ss ** 2).sum())

print(f"B = {B}, quiet final states n = {n}.  Coordinates in units of M_t.  'scores' = tape scores of ages >= 2 (indices 0..7) together.")
print("\n(0) WAVE CLIPS by context of the step: covariance share in the scores, std ratio, share of quiet-state steps with any wave clip in that context")
wany = (Aq[:, :, WAVE_CH] > 0).any(2)
for i, nm in enumerate(CTX):
    m = np.zeros((n, 10, NCH)); m[:, :, WAVE_CH] = (ctx[q] == i)[:, :, None]
    sh = share(rebuild(Aq, 9, m) / Mq)
    print(f"    {nm:38s} share {sh[1]:7.3f}  std ratio {sh[2]:5.2f}   steps in this context {np.mean(ctx[q] == i):.3f}, of which clipped {np.mean(wany[ctx[q] == i]):.2f}")

print("\n(1) SHARE OF THE VARIANCE by class of the causing input (covariance share; columns: 7 coordinates, then scores, then std ratio in the scores)")
print("    'present' = share of quiet final states that have such an input (for spont: such a clip)")
print(f"    {'class':7s} present " + " ".join(f"{c:>9s}" for c in CNAMES) + "    scores  std ratio")
parts = {}
for i, nm in enumerate(CLS):
    parts[nm] = rebuild(Aq, 9, wcls[q] == i) / Mq
    sh = share(parts[nm]); pres = np.mean((np.abs(parts[nm]) > 1e-12).any(1))
    print(f"    {nm:7s} {pres:6.3f}  " + " ".join(f"{v:9.3f}" for v in sh[0]) + f"   {sh[1]:7.3f}   {sh[2]:6.2f}")
print("    mean of each part (units of M):")
for nm in CLS:
    print(f"    {nm:7s}         " + " ".join(f"{v:9.2f}" for v in parts[nm].mean(0)))

print("\n(2) INSIDE the classes M, S>M, S<M: covariance share in the scores by kind of write (the kinds cancel inside a class)")
print(f"    {'class':7s} " + " ".join(f"{g:>16s}" for g in GROUPS) + "   | std ratio of each kind")
for i, nm in enumerate(CLS[:5]):
    row, row2 = [], []
    for g, chs in GROUPS.items():
        m = np.zeros((n, 10, NCH), bool); m[:, :, chs] = (wcls[q] == i)[:, :, chs]
        sh = share(rebuild(Aq, 9, m) / Mq); row.append(sh[1]); row2.append(sh[2])
    print(f"    {nm:7s} " + " ".join(f"{v:16.3f}" for v in row) + "   | " + " ".join(f"{v:5.2f}" for v in row2))

print("\n(3) SPLIT BY CONFIGURATION: covariance share in the scores by class, separately for runner-up after / before the maximum")
for nm, sel in (("runner-up after the maximum", (pS > pM)[q]), ("runner-up before the maximum", (pS < pM)[q])):
    cc = sq[sel] - sq[sel].mean(0); tot = (cc ** 2).sum()
    row = []
    for c in CLS:
        pc = (parts[c][sel] - parts[c][sel].mean(0)) @ R_SCORE[:8].T
        row.append((pc * cc).sum() / tot)
    print(f"    {nm:30s} n = {sel.sum():6d}: " + "  ".join(f"{c} {v:6.3f}" for c, v in zip(CLS, row)))

print("\n(4) LEAVE-OUT TEST: drop all writes caused by one class, rebuild the tape, set it in h_9 (quiet final states)")
print("    changed = share of answers that change (all quiet states / only those where the class is present); acc = accuracy after")
print(f"    baseline accuracy {np.mean(ans0[q] == corr[q]):.4f}")
a0 = ans0[q]; cr = corr[q]; hq = h9[q]
def test(keep, label, pres=None):
    a = answer(set_tape(hq, rebuild(Aq, 9, keep)))
    ch = a != a0
    extra = f" / {np.mean(ch[pres]):.4f} (present in {np.mean(pres):.3f})" if pres is not None else ""
    print(f"    {label:44s} changed {np.mean(ch):.4f}{extra}   acc {np.mean(a == cr):.4f}")
    return a
for i, nm in enumerate(CLS):
    test(wcls[q] != i, "drop " + nm, (np.abs(parts[nm]) > 1e-12).any(1))
test(np.isin(wcls[q], [0, 1, 2]), "keep only M and S (either order)")
test(np.isin(wcls[q], [0, 1, 2, 8]), "keep only M, S, spont")
test(np.isin(wcls[q], [0, 1, 2, 3, 4]), "keep M, S, exmax, exrun")
test(np.isin(wcls[q], [0, 1, 2, 3, 4, 8]), "keep M, S, exmax, exrun, spont")
test(np.isin(wcls[q], [0, 1, 2, 3, 4, 5, 8]), "keep all but far and neg")
test(~np.isin(wcls[q], [6, 7, 8]), "keep all but far, neg, spont")
a = test(np.isin(wcls[q], [0, 1, 2]), "(repeat) keep only M and S: where does the answer go?")
print(f"       of the changed answers: to the maximum {np.mean(a[a != a0] == pM[q][a != a0]):.3f}, to the third {np.mean(a[a != a0] == st['pT'][q][a != a0]):.3f}")

print("\n(5) R2 per coordinate of the tape rebuilt from a subset of classes (quiet final states, coordinates / M)")
for lab, keep in (("M and S only", [0, 1, 2]), ("M, S, exmax, exrun", [0, 1, 2, 3, 4]), ("all but far, neg", [0, 1, 2, 3, 4, 5, 8]), ("all but far, neg, spont", [0, 1, 2, 3, 4, 5])):
    ch = rebuild(Aq, 9, np.isin(wcls[q], keep)) / Mq
    print(f"    {lab:26s} " + " ".join(f"{v:7.3f}" for v in r2(cq, ch)) + f"   scores {1 - (((ch - cq) @ R_SCORE[:8].T) ** 2).sum() / ((sq - sq.mean(0)) ** 2).sum():.3f}")
