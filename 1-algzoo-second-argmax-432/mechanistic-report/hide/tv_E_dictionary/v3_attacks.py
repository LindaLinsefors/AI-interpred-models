"""V3. Attacks on the interpretation "the tape is nearly a function of the candidates" (quiet final states, t = 9, 200000 iid N(0,1)
sequences, seed 5151).  c = 7 tape coordinates / M.
A1  RESAMPLING instead of the filler 0: the inputs outside the named set V are redrawn from N(0,1) conditioned on leaving V unchanged
    (below S for V = (M, S); below the third for V = top 3; below the candidate threshold of their arrival for V = candidates).
    R2_pair = 1 - |c' - c|^2 / |c - mean|^2;  eta2 = 1 - (1 - R2_pair) / 2 = share of the variance of c that V explains;
    changed = share of answers changed when the redrawn sequence's tape replaces the real one.
A2  The functional test split by the gap between runner-up and third, and what the changed answer points at.
A3  Is this special to the tape?  The same skeletons scored on the tape piece and on the rest piece of the scores (entries of age >= 2).
A4  Candidates or simply the positive inputs?  Leave one class of inputs out of the full sequence (set to 0).
A5  Robustness of the R2 to the 1/M normalisation.
Run:  python -B v3_attacks.py > v3_attacks.txt"""
from scipy.special import ndtr, ndtri
from v_common import *

B, t = 200_000, 9
rng = np.random.default_rng(5151)
x = sample(B, 5151); H = run(x)[:, 9]; st = top3(x, 9); q = st["quiet"]; M = st["M"]; ans0 = answer(H); z = zc(H)
prevM, prevS = prev_top2(x); pos = np.arange(10); kw = LEFTM @ w[ORD]
c = to7(z[q]) / M[q, None]; SST = ((c - c.mean(0)) ** 2).sum()
R2 = lambda zf: 1 - ((to7(zf[q] - z[q]) / M[q, None]) ** 2).sum() / SST
changed = lambda zf, s=None: np.mean(answer(set_tape(H[q if s is None else s], zf[q if s is None else s])) != ans0[q if s is None else s])
tapeof = lambda X: zc(run(X)[:, 9])
leak = lambda keep: (np.where(keep, 0.0, x)[:, :, None] * (kw * LAMV ** (9 - pos)[:, None])[None]).sum(1)
isM, isS = pos == st["pM"][:, None], pos == st["pS"][:, None]
rank = np.argsort(np.argsort(-x, 1), 1)
top2arr = x > prevS; cand = top2arr | (x > 0.75 * np.maximum(prevS, 0))
thr_c = np.where(prevS > 0, 0.75 * prevS, prevS)                      # a non-candidate stays one iff x <= thr_c
old = pos < 8

print(f"{q.sum()} quiet final states of {B}.   mean tape: changed {changed(from7(c.mean(0))[None] * M[:, None]):.3f}")
print("\nA1. RESAMPLING the inputs outside V (two independent redraws; the second column pair keeps the real inputs of age 0, 1)")
print("    V                 | filler 0: R2, changed | redraw all others: R2_pair, eta2, changed | redraw only others of age >= 2: R2_pair, eta2, changed | V unchanged")
for name, keep, thr in (("M, S", isM | isS, np.broadcast_to(st["S"][:, None], x.shape)), ("top 3", rank < 3, np.broadcast_to(st["T3"][:, None], x.shape)),
                        ("candidates", cand, thr_c), ("candidates + leak", cand, thr_c)):
    z0 = tapeof(np.where(keep, x, 0.0)) + (leak(keep) if "leak" in name else 0)
    out = []
    for only_old in (False, True):
        redraw = ~keep & (old if only_old else True)
        xr = np.where(redraw, ndtri(rng.random(x.shape) * ndtr(np.where(redraw, thr, 0.0))), x)
        if name == "M, S": same = np.array_equal(top3(xr, 9)["pM"][q], st["pM"][q]) and np.array_equal(top3(xr, 9)["pS"][q], st["pS"][q])
        elif name == "top 3": same = np.array_equal(np.argsort(np.argsort(-xr, 1), 1)[q] < 3, (rank < 3)[q])
        else:
            pM2, pS2 = prev_top2(xr); same = np.array_equal(((xr > pS2) | (xr > 0.75 * np.maximum(pS2, 0)))[q], cand[q])
        zr = tapeof(xr); r = R2(zr); out.append(f"{r:6.3f} {1 - (1 - r) / 2:6.3f} {changed(zr):.3f} [{same}]")
    print(f"    {name:17s} | {R2(z0):6.3f} {changed(z0):.3f} | " + " | ".join(out))

print("\nA2. Functional test split by the gap (S - T) / M between runner-up and third (T = third largest of all ten inputs)")
gap = (st["S"] - st["T3"]) / M; right = ans0 == st["pS"]
subs = {"mean tape": from7(c.mean(0))[None] * M[:, None], "skeleton M, S": tapeof(np.where(isM | isS, x, 0.0)), "skeleton top 3": tapeof(np.where(rank < 3, x, 0.0)),
        "candidates": tapeof(np.where(cand, x, 0.0)), "candidates + leak": tapeof(np.where(cand, x, 0.0)) + leak(cand)}
edges = [0, 0.05, 0.1, 0.2, 0.4, np.inf]
print("    gap bin        share   network right | changed: " + " | ".join(subs))
for a, b in zip(edges[:-1], edges[1:]):
    s = q & (gap >= a) & (gap < b)
    print(f"    [{a:4.2f}, {b:4.2f})   {s.sum() / q.sum():.3f}   {right[s].mean():.3f}         | " + " | ".join(f"{changed(zf, s):9.3f}" for zf in subs.values()))
print("    where the changed answers point (share of all quiet states): to the third | to the maximum | to an input of age 0 or 1 | elsewhere;   and: answer right before -> wrong after | wrong -> right")
for n, zf in subs.items():
    a1 = answer(set_tape(H[q], zf[q])); a0 = ans0[q]; ch = a1 != a0
    print(f"    {n:18s} {np.mean(ch & (a1 == st['pT'][q])):.3f} | {np.mean(ch & (a1 == st['pM'][q])):.3f} | {np.mean(ch & (a1 >= 8)):.3f} | "
          f"{np.mean(ch & (a1 != st['pT'][q]) & (a1 != st['pM'][q]) & (a1 < 8)):.3f}   ;   {np.mean((a0 == st['pS'][q]) & (a1 != st['pS'][q])):.3f} | {np.mean((a0 != st['pS'][q]) & (a1 == st['pS'][q])):.3f}")
print("    split by whether an input of age 0 or 1 is the third or better (it cannot be M or S in a quiet state):")
newT = st["pT"] >= 8
for lab, s in (("third has age >= 2", q & ~newT), ("third has age 0 or 1", q & newT)):
    print(f"    {lab:22s} share {s.sum() / q.sum():.3f} | changed: " + " | ".join(f"{n} {changed(zf, s):.3f}" for n, zf in subs.items()))

print("\nA3. Is it special to the tape?  Skeleton states scored on three pieces: R2 of the tape coordinates; R2 of the tape piece of the scores")
print("    (entries of age >= 2, each minus the mean over those 8 entries); R2 of the REST piece of the scores (same entries) and of the three pulse units.")
def pieces(Hh):
    s_t = Hh[:, ORD] @ P_TAPE.T @ UO.T; s_r = Hh @ U.T - s_t
    cen = lambda s: (s[:, :8] - s[:, :8].mean(1, keepdims=True)) / M[:, None]
    return cen(s_t)[q], cen(s_r)[q], (Hh[:, PULSE] / M[:, None])[q]
r2g = lambda a, b: 1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
real = pieces(H)
print(f"    rms over entries of the centred pieces / M: tape piece {np.sqrt((real[0] ** 2).mean()):.2f}, rest piece {np.sqrt((real[1] ** 2).mean()):.2f}")
for n, keep in (("M, S", isM | isS), ("top 3", rank < 3), ("top 4", rank < 4), ("candidates", cand), ("all positive inputs", x > 0)):
    Hs = run(np.where(keep, x, 0.0))[:, 9]; sk = pieces(Hs)
    print(f"    {n:20s} tape coordinates {R2(zc(Hs)):.3f} | tape piece of the scores {r2g(real[0], sk[0]):.3f} | rest piece of the scores {r2g(real[1], sk[1]):.3f} | pulse units {r2g(real[2], sk[2]):.3f}")

print("\nA4. Candidates or the positive inputs?  One class of inputs set to 0 in the otherwise real sequence (ages >= 2 only, so that the newest inputs' direct write is not involved)")
print("    class                                               inputs per state | states with any | R2 (all quiet) | R2 on the states with any | changed on the states with any | the same with the linear leak added back")
top4 = rank < 4
classes = {"positive, not a candidate": (x > 0) & ~cand, "negative candidate": (x <= 0) & cand, "candidate, not in the top 4": cand & ~top4,
           "in the top 4, not a candidate": top4 & ~cand, "candidate below the runner-up of its arrival ('near')": cand & ~top2arr,
           "dethroned (top two at arrival, not M or S now)": top2arr & ~isM & ~isS, "negative, not a candidate": (x <= 0) & ~cand}
for n, cls in classes.items():
    cls = cls & old; anyc = q & cls.any(1); keep = ~cls
    zf = tapeof(np.where(keep, x, 0.0)); zl = zf + leak(keep)
    r2s = lambda zz: 1 - ((to7(zz[anyc] - z[anyc]) / M[anyc, None]) ** 2).sum() / ((to7(z[anyc]) / M[anyc, None] - c.mean(0)) ** 2).sum()
    print(f"    {n:52s} {cls[q].sum(1).mean():5.2f} | {anyc.sum() / q.sum():.3f} | {R2(zf):.3f} | {r2s(zf):.3f} | {changed(zf, anyc):.3f} | {r2s(zl):.3f} {changed(zl, anyc):.3f}")

print("\nA5. Robustness of R2 to the normalisation by M (skeletons with filler 0)")
big = q & (M > np.median(M[q]))
for n, keep in (("M, S", isM | isS), ("top 3", rank < 3), ("candidates", cand)):
    zf = tapeof(np.where(keep, x, 0.0)); d = to7(zf - z)
    raw = 1 - (d[q] ** 2).sum() / ((to7(z[q]) - to7(z[q]).mean(0)) ** 2).sum()
    rb = 1 - ((d[big] / M[big, None]) ** 2).sum() / ((to7(z[big]) / M[big, None] - (to7(z[big]) / M[big, None]).mean(0)) ** 2).sum()
    cm = np.median(np.linalg.norm(d[q], axis=1) / np.linalg.norm(to7(z[q]), axis=1))
    print(f"    {n:12s} R2 of c / M {R2(zf):.3f} | R2 without dividing by M {raw:.3f} | R2 of c / M on states with M above its median {rb:.3f} | median over states of |error| / |tape| {cm:.3f}, "
          f"90th percentile {np.quantile(np.linalg.norm(d[q], axis=1) / np.linalg.norm(to7(z[q]), axis=1), .9):.3f}")
