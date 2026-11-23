"""v1: reproduction of the position claims (1, 3, 6, 7, 9) of t1_transplants with own code and other seeds.

Pairs: A = quiet state at step t;  B = A with one thing changed (still quiet).  Hybrid "fw" = rest of A + tape of B.
  S_pos   runner-up swapped with a far-below slot (age >= 2, not top three, value < S - 0.4 M)
  M_pos   maximum swapped with such a slot
  low_val every far-below input redrawn from N(0,1) truncated at S - 0.4 M;   low_pos: those inputs permuted
  unrel   B = another quiet sequence rescaled to A's M
"""
from scipy.stats import norm
from v_common import *

N = 500_000
ar_ = lambda n: np.arange(n)
top8 = lambda s: s[:, :8].argmax(1)

# ---------------------------------------------------------------- claim 3: the two score pieces
x, st = quiet(N, 9, 9101)
h = state(x, 9)
s_all = h @ U.T
s_tape = h[:, O] @ P_T.T @ U[:, O].T
s_rest = s_all - s_tape
n = len(x)
print(f"[claim 3] top entry among ages >= 2, {n} quiet final states of {N}")
for lab, s in (("full", s_all), ("tape piece", s_tape), ("rest piece", s_rest)):
    a = top8(s)
    print(f"   {lab:11s} runner-up {np.mean(a == st['pS']):.3f}  maximum {np.mean(a == st['pM']):.3f}  third {np.mean(a == st['pT']):.3f}")
    dSM = (s[ar_(n), st["pS"]] - s[ar_(n), st["pM"]]) / st["M"]
    ok = st["aT"] >= 2
    dST = ((s[ar_(n), st["pS"]] - s[ar_(n), st["pT"]]) / st["M"])[ok]
    q = lambda v: " ".join(f"{u:6.1f}" for u in np.quantile(v, [.1, .25, .5, .75, .9]))
    print(f"   {'':11s} S entry - M entry (in M), 10/25/50/75/90%: {q(dSM)}  >0: {np.mean(dSM > 0):.3f}"
          f"   S entry - T entry: {q(dST)}  >0: {np.mean(dST > 0):.3f}")
print("   by a_S: P(runner-up on top among ages >= 2)   a_S:    " + "".join(f"{a:7d}" for a in range(2, 10)))
for lab, s in (("tape piece", s_tape), ("rest piece", s_rest)):
    hit = top8(s) == st["pS"]
    print(f"   {lab:44s}" + "".join(f"{hit[st['aS'] == a].mean():7.3f}" for a in range(2, 10)))

# ---------------------------------------------------------------- claims 1, 9: S_pos, read out at once
for t in (9, 6):
    x, st = quiet(N, t, 9200 + t)
    rng = np.random.default_rng(9300 + t)
    hA = state(x, t)
    j, ok = pick_low_slot(x, st, t, rng)
    xB = swapped(x, st["pS"], j)
    sB = stats(xB, t)
    ok &= sB["quiet"]
    a, b, hA_, hB_ = sub(st, ok), sub(sB, ok), hA[ok], state(xB[ok], t)
    fw, rv = hyb(hA_, hB_), hyb(hB_, hA_)
    names = {"tape's S": b["pS"], "rest's S": a["pS"], "M": a["pM"], "third(A)": a["pT"]}
    neg = (fw < -1e-9).any(1)
    print(f"\n[claim 1] S_pos, t = {t}, n = {ok.sum()} pairs; intact A names its runner-up in {np.mean(ans_at(hA_, t) == a['pS']):.3f}")
    print(f"   rest A + tape B: {shares(ans_at(fw, t), names)}")
    print(f"   clipped        : {shares(ans_at(fw, t, True), names)}")
    print(f"   rest B + tape A: tape's S {np.mean(ans_at(rv, t) == a['pS']):.3f}  rest's S {np.mean(ans_at(rv, t) == b['pS']):.3f}")
    hit = ans_at(fw, t) == b["pS"]
    print(f"   [claim 9] negative entry in {neg.mean():.3f} of hybrids; tape's S named in {hit[~neg].mean():.3f} without / {hit[neg].mean():.3f} with one;"
          f" most negative entry / M among the latter: median {np.median((fw.min(1) / a['M'])[neg]):.2f}, 10% quantile {np.quantile((fw.min(1) / a['M'])[neg], .1):.2f};"
          f" rms activation of intact states / M: {np.sqrt(np.mean((hA_ / a['M'][:, None]) ** 2)):.1f}")
    near = a["T"] > a["S"] - 0.1 * a["M"]
    print(f"   third within 0.1 M of S (n = {near.sum()}): tape's S {hit[near].mean():.3f}, third {np.mean(ans_at(fw, t)[near] == a['pT'][near]):.3f};"
          f"   further away: tape's S {hit[~near].mean():.3f}")
    if t == 9:
        print("   by donor's a_S:        " + "  ".join(f"{s}: {hit[b['aS'] == s].mean():.3f}" for s in range(2, 10)))
        print("   by recipient's a_S:    " + "  ".join(f"{s}: {hit[a['aS'] == s].mean():.3f}" for s in range(2, 10)))
        m38 = (b["aS"] >= 3) & (b["aS"] <= 8)
        print(f"   donor 3 <= a_S <= 8 (n = {m38.sum()}): tape's {hit[m38].mean():.3f}, rest's {np.mean(ans_at(fw, t)[m38] == a['pS'][m38]):.4f}")
        for lab, m in (("donor a_S = 2, a_M <= 7", (b["aS"] == 2) & (b["aM"] <= 7)), ("donor a_S = 2, a_M = 8", (b["aS"] == 2) & (b["aM"] == 8)),
                       ("donor a_S = 2, a_M = 9", (b["aS"] == 2) & (b["aM"] == 9))):
            print(f"   {lab} (n = {m.sum()}): tape's {hit[m].mean():.3f}, maximum {np.mean(ans_at(fw, t)[m] == a['pM'][m]):.3f}")
        cells = np.array([hit[(b["aM"] == mm) & (b["aS"] == ss)].mean() for mm in range(2, 10) for ss in range(2, 10) if mm != ss])
        print(f"   56 cells of the donor's (a_M, a_S): median {np.median(cells):.3f}, range {cells.min():.3f} to {cells.max():.3f}")

# ---------------------------------------------------------------- claim 1: S_pos, then low inputs
for T0 in (6, 4, 8):
    x, st = quiet(N, T0, 9400 + T0)
    rng = np.random.default_rng(9500 + T0)
    j, ok = pick_low_slot(x, st, T0, rng)
    xB = swapped(x, st["pS"], j)
    ok &= stats(xB, T0)["quiet"]
    a, b = sub(st, ok), sub(stats(xB, T0), ok)
    hA_, hB_ = state(x[ok], T0), state(xB[ok], T0)
    fut = lows(a["M"], 9 - T0, rng)
    fw = hyb(hA_, hB_)
    f, fc = final(fw, fut), final(fw, fut, True)
    print(f"\n[claim 1] S_pos at t = {T0}, then {9 - T0} low inputs, n = {ok.sum()}: tape's S {np.mean(f == b['pS']):.3f}, rest's S {np.mean(f == a['pS']):.3f}"
          f"   clipped: {np.mean(fc == b['pS']):.3f}, {np.mean(fc == a['pS']):.3f};  intact A correct {np.mean(final(hA_, fut) == a['pS']):.3f}")
    if T0 == 6:
        hit = f == b["pS"]
        print("   by donor's a_S at t = 6: " + "  ".join(f"{s}: {hit[b['aS'] == s].mean():.3f}" for s in range(2, 7)))

# ---------------------------------------------------------------- claims 1, 3: unrelated donor
for t in (9, 6):
    x, st = quiet(N, t, 9600 + t)
    rng = np.random.default_rng(9700 + t)
    perm = rng.permutation(len(x))
    xB = x[perm] * (st["M"] / st["M"][perm])[:, None]
    sB = stats(xB, t)
    fw = hyb(state(x, t), state(xB, t))
    an = ans_at(fw, t)
    d = st["pS"] != sB["pS"]
    print(f"\n[claim 1] unrelated donor rescaled to A's M, t = {t}: runner-up positions differ in n = {d.sum()}: tape's S {np.mean(an[d] == sB['pS'][d]):.3f},"
          f" rest's S {np.mean(an[d] == st['pS'][d]):.3f};  clipped: {np.mean(ans_at(fw, t, True)[d] == sB['pS'][d]):.3f}")
    c = sB["pS"] == st["pM"]
    print(f"   [claim 3] tape's runner-up sits on the rest's maximum (n = {c.sum()}): hybrid names that position in {np.mean(an[c] == sB['pS'][c]):.3f}")

# ---------------------------------------------------------------- claim 7: far-below inputs (+ control)
for t in (9, 6):
    x, st = quiet(N, t, 9800 + t)
    rng = np.random.default_rng(9900 + t)
    hA = state(x, t)
    low = far_low(x, st, t)
    pre = x[:, : t + 1]
    # low_val
    new = norm.ppf(rng.random(low.shape) * norm.cdf(st["S"] - 0.4 * st["M"])[:, None])
    xB = x.copy(); xB[:, : t + 1] = np.where(low, new, pre)
    ok = low.any(1)
    # low_pos
    cnt = low.sum(1)
    asc = np.argsort(np.where(low, np.arange(t + 1)[None], 99), 1)
    shu = np.argsort(np.where(low, rng.random(low.shape), 99), 1)
    k = np.arange(t + 1)[None] < cnt[:, None]
    newp = pre.copy()
    np.put_along_axis(newp, asc, np.where(k, np.take_along_axis(pre, shu, 1), np.take_along_axis(pre, asc, 1)), 1)
    xP = x.copy(); xP[:, : t + 1] = newp
    okP = (newp != pre).any(1)
    assert np.allclose(np.sort(newp, 1), np.sort(pre, 1)) and np.array_equal(newp[~low], pre[~low])
    for lab, xb, m in (("low_val", xB, ok), ("low_pos", xP, okP)):
        sb = stats(xb, t)
        assert np.array_equal(sb["pS"][m], st["pS"][m]) and np.array_equal(sb["pM"][m], st["pM"][m])
        hB = state(xb, t)
        dT = part(hB) - part(hA)
        rel = np.linalg.norm(dT[m], axis=1) / np.linalg.norm(part(hA)[m], axis=1)
        fw = hA + dT
        perm = rng.permutation(len(x))
        ctl = hA + dT[perm] * (st["M"] / st["M"][perm])[:, None]          # same-sized change from another pair
        relc = np.linalg.norm((dT[perm] * (st["M"] / st["M"][perm])[:, None])[m & m[perm]], axis=1) / np.linalg.norm(part(hA)[m & m[perm]], axis=1)
        acc = lambda hh, mm: np.mean(ans_at(hh[mm], t) == st["pS"][mm])
        print(f"\n[claim 7] {lab}, t = {t}, n = {m.sum()}: tape changes by median {np.median(rel):.2f} of its norm (quartiles {np.quantile(rel, .25):.2f}, {np.quantile(rel, .75):.2f});"
              f" accuracy intact {acc(hA, m):.3f}, rest A + tape B {acc(fw, m):.3f}")
        print(f"   CONTROL (attack 2): the tape change of ANOTHER pair, rescaled to this M (median relative size {np.median(relc):.2f}), n = {(m & m[perm]).sum()}:"
              f" accuracy {acc(ctl, m & m[perm]):.3f}")
        if t == 6:
            fut = lows(st["M"], 3, rng)
            for fl, kpos, val in (("3 low", None, None), ("new max at 7", 0, 1.3 * st["M"]), ("new max at 9", 2, 1.3 * st["M"])):
                f = fut.copy()
                if kpos is not None:
                    f[:, kpos] = val
                truth = np.argsort(np.concatenate([pre, f], 1), 1)[:, -2]
                print(f"   then {fl:13s}: intact {np.mean(final(hA[m], f[m]) == truth[m]):.3f}, hybrid {np.mean(final(fw[m], f[m]) == truth[m]):.3f},"
                      f" control {np.mean(final(ctl[m & m[perm]], f[m & m[perm]]) == truth[m & m[perm]]):.3f}")
    # everything except the top two redrawn below S (the top two keep value and position)
    top2 = np.zeros(pre.shape, bool); top2[ar_(len(x)), st["pM"]] = True; top2[ar_(len(x)), st["pS"]] = True
    new = norm.ppf(rng.random(pre.shape) * norm.cdf(st["S"])[:, None])
    xb = x.copy(); xb[:, : t + 1] = np.where(top2, pre, new)
    fw = hyb(hA, state(xb, t))
    print(f"[claim 7] t = {t}: all inputs except the top two redrawn below S in the donor: hybrid accuracy {np.mean(ans_at(fw, t) == st['pS']):.3f} (intact {np.mean(ans_at(hA, t) == st['pS']):.3f})")

# ---------------------------------------------------------------- claim 6: position of the maximum, then a new maximum
T0 = 6
x, st = quiet(N, T0, 9951)
rng = np.random.default_rng(9952)
j, ok = pick_low_slot(x, st, T0, rng)
xB = swapped(x, st["pM"], j)
ok &= stats(xB, T0)["quiet"]
a, b = sub(st, ok), sub(stats(xB, T0), ok)
hA_, hB_ = state(x[ok], T0), state(xB[ok], T0)
fw = hyb(hA_, hB_)
print(f"\n[claim 6] M_pos at t = 6, n = {ok.sum()};  present answer of the hybrid: runner-up {np.mean(ans_at(fw, T0) == a['pS']):.3f}, tape's maximum {np.mean(ans_at(fw, T0) == b['pM']):.3f}")
for lab, kpos in (("new maximum 1.3 M at step 7, then two low", 0), ("two low, new maximum at step 9", 2)):
    f = lows(a["M"], 3, rng); f[:, kpos] = 1.3 * a["M"]
    names = {"rest's old max": a["pM"], "tape's old max": b["pM"], "old runner-up": a["pS"], "new max": np.full(ok.sum(), T0 + 1 + kpos)}
    print(f"   {lab}: intact {np.mean(final(hA_, f) == a['pM']):.3f};  hybrid: {shares(final(fw, f), names)}")
    print(f"   {'':42s} clipped: {shares(final(fw, f, True), names)}")
    gap = (a["M"] - a["S"]) / a["M"]
    an = final(fw, f)
    print("      by donor's a_M: rest's / tape's   " + "  ".join(f"{m}: {np.mean(an[b['aM'] == m] == a['pM'][b['aM'] == m]):.2f}/{np.mean(an[b['aM'] == m] == b['pM'][b['aM'] == m]):.2f}" for m in range(2, 7)))
    print("      by recipient's a_M: rest's / tape's " + "  ".join(f"{m}: {np.mean(an[a['aM'] == m] == a['pM'][a['aM'] == m]):.2f}/{np.mean(an[a['aM'] == m] == b['pM'][a['aM'] == m]):.2f}" for m in range(2, 7)))
