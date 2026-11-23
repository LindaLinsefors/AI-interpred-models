"""D3b. Which inputs does the tape of a quiet final state (t = 9) depend on?  More skeletons, and N1 by number of later writes.
Skeleton = tape of the controlled sequence that keeps a named set of inputs and replaces the others by 0; the tape part of the real
h_9 is replaced by it (set_tape) and the answer compared.  B = 200000 sequences, seed 2312, iid N(0,1).
"Top two at arrival": x_p > runner-up of x_0..x_(p-1) (the first two inputs always qualify).
Run:  python -B d3b_which_inputs.py > d3b_which_inputs.txt"""
from dict_common import *

B, t = 200_000, 9
x = sample(B, 2312); H = run(x); st = running_stats(x); ps = prefix_stats(x, t); q = ps["quiet"]; Mq = ps["M"][q, None]
ans0 = answer(H[:, 9]); zreal = zc(H[:, 9]); c = to7(zreal[q]) / Mq; pos = np.arange(10)
kAc = from7(np.load(HERE / "d1_patterns.npz")["kA"])
prevM = np.concatenate([np.full((B, 1), -np.inf), st["M1"][:, :-1]], 1); prevS = np.concatenate([np.full((B, 1), -np.inf), st["M2"][:, :-1]], 1)
isM, isS = pos == ps["pM"][:, None], pos == ps["pS"][:, None]
top2arr, rec = x > prevS, x > prevM
near = x > 0.75 * np.maximum(prevS, 0)
rank = np.argsort(np.argsort(-x, 1), 1)
skel = lambda keep: zc(run(np.where(keep, x, 0.0))[:, -1])
R2 = lambda zf, s=None: (lambda s_: 1 - ((to7(zf[s_]) / ps["M"][s_, None] - to7(zreal[s_]) / ps["M"][s_, None]) ** 2).sum()
                         / ((to7(zreal[s_]) / ps["M"][s_, None] - c.mean(0)) ** 2).sum())(q if s is None else s)
chg = lambda zf, s=None: (lambda s_: np.mean(answer(set_tape(H[s_, 9], to7(zf[s_]))) != ans0[s_]))(q if s is None else s)

print(f"{q.sum()} quiet final states of {B}.  Columns: inputs kept per state (mean), R2 of all 7 coordinates / M, share of answers changed.")
sets = {"M, S": isM | isS, "top 3": rank < 3, "top 4": rank < 4, "top 5": rank < 5, "all positive inputs": x > 0,
        "M, S + every maximum-at-arrival (records)": isM | isS | rec,
        "every input in the top two at its arrival": top2arr,
        "top two at arrival, or above 0.75 of the runner-up then": top2arr | near,
        "top two at arrival + positive inputs of age 0, 1": top2arr | ((pos >= 8) & (x > 0)),
        "top two at arrival + all inputs of age 0, 1": top2arr | (pos >= 8),
        "candidates (line 8) + all inputs of age 0, 1": top2arr | near | (pos >= 8),
        "all but the inputs of age 0, 1": np.broadcast_to(pos < 8, x.shape), "all but the input of age 0": np.broadcast_to(pos < 9, x.shape),
        "top 3 + inputs of age 0, 1": (rank < 3) | (pos >= 8)}
Z = {}
kw = LEFTM @ w[ORD]
leak = lambda keep: (np.where(keep, 0.0, x)[:, :, None] * (kw * LAMV ** (9 - pos)[:, None])[None]).sum(1)   # direct write of the dropped inputs
for n, keep in sets.items():
    Z[n] = skel(keep); zl = Z[n] + leak(keep)
    print(f"   {n:58s} kept {keep[q].sum(1).mean():5.2f}   R2 {R2(Z[n]):6.3f}   answers changed {chg(Z[n]):.3f}"
          f"   | + linear leak of the dropped inputs: R2 {R2(zl):6.3f}   answers changed {chg(zl):.3f}")
print("   (linear leak = sum over dropped inputs of (l.w[o]) x_p lam^age, the direct write of an input that moves no gate)")

print("\nThe same without the newest entries: is the runner-up still the top score among the entries of age >= 2 (indices 0..7)?")
top8 = lambda h: (h @ U.T)[:, :8].argmax(1)
print(f"   real states: {np.mean(top8(H[q, 9]) == ps['pS'][q]):.3f}")
for n in ("M, S", "top 3", "top 4", "every input in the top two at its arrival", "top two at arrival, or above 0.75 of the runner-up then", "all positive inputs"):
    hm = set_tape(H[q, 9], to7(Z[n][q]))
    print(f"   {n:58s} runner-up on top {np.mean(top8(hm) == ps['pS'][q]):.3f}   top entry changed {np.mean(top8(hm) != top8(H[q, 9])):.3f}")

print("\nFormula N1 (skeleton (M, S) + g (x_p - Theta_p)_+ kA lam^(age-2) for inputs after both), by the number of later inputs it writes (age >= 2):")
start = np.maximum(ps["pM"], ps["pS"]) + 1; Th = 0.75 * ps["S"]; nw = np.zeros(B); zf = Z["M, S"].copy()
for p in range(10):
    e = np.where(p >= start, np.maximum(x[:, p] - Th, 0), 0.0)
    if t - p >= 2: zf = zf + (19.1 * e)[:, None] * kAc * LAMV ** (t - p - 2); nw += e > 0
    Th = Th + 0.44 * e
after = skel(isM | isS | ((pos > np.maximum(ps["pM"], ps["pS"])[:, None]) & (pos < 8)))
print("   'exact' = skeleton that keeps M, S and the real inputs after both with age >= 2")
for k in (0, 1, 2):
    s = q & (nw == k if k < 2 else nw >= 2)
    print(f"   {k}{'+' if k == 2 else ' '} writes (n = {s.sum():6d}):  R2 skeleton (M, S) {R2(Z['M, S'], s):.3f}, N1 {R2(zf, s):.3f}, exact {R2(after, s):.3f} | "
          f"answers changed: {chg(Z['M, S'], s):.3f}, {chg(zf, s):.3f}, {chg(after, s):.3f}")
    d_true, d_form = to7(after[s] - Z["M, S"][s]), to7(zf[s] - Z["M, S"][s])
    if k: print(f"        the later inputs' own contribution (exact skeleton minus skeleton (M, S)): R2 of N1's version {1 - ((d_true - d_form) ** 2).sum() / (d_true ** 2).sum():.3f} (uncentred)")
