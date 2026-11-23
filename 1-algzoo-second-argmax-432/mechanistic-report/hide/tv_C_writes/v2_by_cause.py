"""V2. Checks of claims 3 and 4 (the tape by the input that caused each write), with three attacks:
 (A) split by the age of the maximum and of the runner-up inside each order (the investigator pooled them);
 (B) the same accounting for the five NON-tape modes (is the result specific to the tape?);
 (C) another attribution: the exact gate-fixed dependence on each input VALUE (d h_9 / d x_p with gates fixed).
Own code, seed 7202.   Run:  python -B v2_by_cause.py > v2_by_cause.txt"""
from vcommon import *

CLS = ["M", "S>M", "S<M", "exmax", "exrun", "near", "far", "neg", "spont"]


def shift(a, n, fill):
    out = np.full_like(a, fill); out[:, n:] = a[:, : a.shape[1] - n]
    return out


def write_classes(x, t):
    """Class of the input that caused each write, seen from step t: (B, t+1, 17). The investigator's rules, re-coded."""
    x = x[:, : t + 1]; B = len(x)
    vals, _ = prefix_before(x); kind = arrival_kind(x, vals); st = top3(x, t)
    jj = np.arange(t + 1)[None]; pM, pS = st["pM"][:, None], st["pS"][:, None]
    role = np.array([7, 3, 4, 5, 6])[kind]
    role[(jj == pS) & (pS > pM)] = 1; role[(jj == pS) & (pS < pM)] = 2; role[jj == pM] = 0
    k1, k2, r1, r2_ = shift(kind, 1, 0), shift(kind, 2, 0), shift(role, 1, 7), shift(role, 2, 7)
    wave = np.select([kind == 1, kind == 2, k1 == 1, k2 == 1, (k2 == 2) | (k2 == 3), (k1 == 2) | (k1 == 3)], [role, role, r1, r2_, r2_, r1], 8)
    wc = np.empty((B, t + 1, NCH), np.int8)
    wc[:, :, 0] = role; wc[:, :, CCH] = role[:, :, None]; wc[:, :, 1] = r1; wc[:, :, 2] = r2_; wc[:, :, 3] = r2_
    wc[:, :, WCH] = wave[:, :, None]
    return wc, st


B = 300_000
x = sample(B, 7202)
H, Z, Amp = amplitudes(x)
wc, st = write_classes(x, 9)
q = st["quiet"]; Mq = st["M"][q, None]; pM, pS = st["pM"][q], st["pS"][q]; after = pS > pM
Aq, wq, hq = Amp[q], wc[q], H[q, 9]; a0 = answer(hq); corr = second_argmax(x)[q]; n = q.sum()
S8 = Uo[:8].T                                                      # tape part -> scores of ages >= 2
whole = rebuild(Aq, 9) / Mq @ S8
parts = {c: rebuild(Aq, 9, wq == i) / Mq @ S8 for i, c in enumerate(CLS)}
print(f"{B} N(0,1) sequences (seed 7202), quiet final states n = {n}, runner-up after the maximum in {after.mean():.3f}; baseline accuracy {np.mean(a0 == corr):.4f}")

print("\n(1) COVARIANCE SHARE by class of the causing input, tape scores of ages >= 2 (std ratio in brackets)")
for nm, sel in (("all quiet final states", np.ones(n, bool)), ("runner-up AFTER the maximum", after), ("runner-up BEFORE the maximum", ~after)):
    print(f"    {nm:30s} " + "  ".join("{} {:+.3f} ({:.2f})".format(c, *cov_share(parts[c][sel], whole[sel])) for c in CLS))

print("\n(2) LEAVE-OUT by class: share of answers changed among quiet final states where the class is present")
for i, c in enumerate(CLS):
    pres = (np.abs(parts[c]) > 1e-12).any(1)
    a = answer(with_tape(hq, rebuild(Aq, 9, wq != i)))
    print(f"    drop {c:6s} changed {np.mean((a != a0)[pres]):.4f}   (present in {pres.mean():.3f})")
for lab, keep in (("M and S only", [0, 1, 2]), ("M, S, exmax, exrun", [0, 1, 2, 3, 4]), ("all but far and neg", [0, 1, 2, 3, 4, 5, 8])):
    t13 = rebuild(Aq, 9, np.isin(wq, keep)); a = answer(with_tape(hq, t13)); ch = a != a0
    r2s = 1 - ((t13 / Mq @ S8 - whole) ** 2).sum() / ((whole - whole.mean(0)) ** 2).sum()
    print(f"    keep {lab:22s} changed {ch.mean():.4f}  (after {ch[after].mean():.4f}, before {ch[~after].mean():.4f})   R2 of the scores {r2s:.3f}"
          f"   changed answers that become the maximum's position: {np.mean(a[ch] == pM[ch]):.3f}")

print("\n(A) SPLIT BY THE AGES: covariance share of M / S / exrun, by the age of the more recent of the two (ages 2..8)")
aM, aS = 9 - pM, 9 - pS
for nm, sel0, ag in (("runner-up AFTER the maximum, by age of the runner-up", after, aS), ("runner-up BEFORE the maximum, by age of the maximum", ~after, aM)):
    print(f"    {nm}")
    for a in range(2, 9):
        sel = sel0 & (ag == a)
        sS = cov_share((parts["S>M"] + parts["S<M"])[sel], whole[sel])[0]
        print(f"      age {a}: n = {sel.sum():6d}   M {cov_share(parts['M'][sel], whole[sel])[0]:+.3f}   S {sS:+.3f}   exrun {cov_share(parts['exrun'][sel], whole[sel])[0]:+.3f}"
              f"   exmax {cov_share(parts['exmax'][sel], whole[sel])[0]:+.3f}   std of the scores {whole[sel].std():.2f}")

print("\n(B) CONTROL: the same accounting for the five NON-tape modes of the 13 non-pulse units (level, runner, clock, trans, echo), scores of ages >= 2")
GR = kernel(PR)
wholeR = rebuild(Aq, 9, G=GR) / Mq @ S8
print(f"    check: writes rebuild the rest part, max error {np.abs(rebuild(Aq, 9, G=GR) - hq[:, ORD] @ PR.T).max():.1e}")
for nm, sel in (("all quiet final states", np.ones(n, bool)), ("runner-up AFTER the maximum", after), ("runner-up BEFORE the maximum", ~after)):
    print(f"    {nm:30s} " + "  ".join("{} {:+.3f}".format(c, cov_share((rebuild(Aq[sel], 9, wq[sel] == i, G=GR) / Mq[sel] @ S8), wholeR[sel])[0]) for i, c in enumerate(CLS)))
print("    by kind of write (std ratio / covariance share), rest modes: " + "  ".join(
    "{} {:.2f} / {:+.2f}".format(k, *cov_share(rebuild(Aq, 9, np.isin(np.arange(NCH), chs).astype(float), G=GR) / Mq @ S8, wholeR)[::-1]) for k, chs in KINDS.items()))

print("\n(C) OTHER ATTRIBUTION: exact dependence on each input VALUE with this sequence's gates fixed, h_9 = sum_p J_p x_p")
nb = 60_000
xs = x[q][:nb]; Hs, Zs = run(xs, return_pre=True); Gt = (Zs > 0).astype(float)
J = np.zeros((nb, 10, 16))
for t in range(10):
    J[:, :t] = (J[:, :t] @ W.T) * Gt[:, t, None, :]
    J[:, t] = Gt[:, t] * w
print(f"    check on {nb} quiet sequences: max |sum_p J_p x_p - h_9| = {np.abs(np.einsum('bpu,bp->bu', J, xs) - Hs[:, 9]).max():.1e}")
contrib = (J[:, :, ORD] @ P.T @ S8) * (xs / Mq[:nb])[:, :, None]       # (nb, 10 inputs, 8 entries)
pT = st["pT"][q][:nb]; jj = np.arange(10)[None]
vrole = np.where(jj == pM[:nb, None], 0, np.where(jj == pS[:nb, None], 1, np.where(jj == pT[:, None], 2, np.where(xs > 0, 3, 4))))
VR = ["value of the maximum", "value of the runner-up", "value of the third", "other positive inputs", "non-positive inputs"]
wh = whole[:nb]; af = after[:nb]
for nm, sel in (("all", np.ones(nb, bool)), ("runner-up AFTER the maximum", af), ("runner-up BEFORE the maximum", ~af)):
    print(f"    {nm:30s} " + "  ".join("{}: {:+.3f} ({:.2f})".format(v, *cov_share((contrib * (vrole == i)[:, :, None]).sum(1)[sel], wh[sel])) for i, v in enumerate(VR)))
print("    the same for the investigator's attribution on these sequences (M / S / everything else): " + "  ".join(
    f"{nm}: M {cov_share(parts['M'][:nb][sel], wh[sel])[0]:+.3f} S {cov_share((parts['S>M'] + parts['S<M'])[:nb][sel], wh[sel])[0]:+.3f}" for nm, sel in (("after", af), ("before", ~af))))

print("\n(D) RUN CONTINUED from quiet states at step 6 (inputs x_7..x_9 as they were): tape rebuilt from the writes of M and S only")
t = 6
wct, stt = write_classes(x, t); qt = stt["quiet"]; ht = H[qt, t]
cont = lambda h: answer(run(x[qt, t + 1:], h0=h)[:, -1])
ref = cont(ht)
for lab, keep in (("M and S only", [0, 1, 2]), ("all but far and neg", [0, 1, 2, 3, 4, 5, 8])):
    hm = with_tape(ht, rebuild(Amp[qt], t, np.isin(wct[qt], keep)))
    print(f"    {lab:22s} n = {qt.sum()}  changed {np.mean(cont(hm) != ref):.4f}   neg {np.mean((hm < -1e-9).any(1)):.3f}   clipped: changed {np.mean(cont(np.maximum(hm, 0)) != ref):.4f}")
print(f"    tape deleted            changed {np.mean(cont(with_tape(ht, 0 * ht[:, ORD])) != ref):.4f}")
