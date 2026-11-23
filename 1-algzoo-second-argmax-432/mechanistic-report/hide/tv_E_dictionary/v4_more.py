"""V4. (a) A smaller set than "candidates": V_f = the inputs with x_p > f max(S_(p-1), 0), S_(p-1) = runner-up of x_0..x_(p-1) (0 if absent).
        Skeleton (filler 0) and resampling test, plus the ACCURACY (answer = true runner-up) after the substitution.  Quiet final states.
    (b) The write threshold of unit 5 in random sequences (the investigator measured it only in controlled ones).
    (c) Recomputation of claim 6 (one more quiet step) and claim 8 (capacity of the dictionary).
Seeds 6161.., iid N(0,1).   Run:  python -B v4_more.py > v4_more.txt"""
from itertools import combinations
from scipy.special import ndtr, ndtri
from v_common import *

rng = np.random.default_rng(6161)
B = 200_000
x = sample(B, 6161); Hall, Zall = run(x, return_pre=True); H = Hall[:, 9]; st = top3(x, 9); q = st["quiet"]; M = st["M"]; ans0 = answer(H); z = zc(H)
prevM, prevS = prev_top2(x); pos = np.arange(10); kw = LEFTM @ w[ORD]; old = pos < 8
c = to7(z[q]) / M[q, None]; SST = ((c - c.mean(0)) ** 2).sum()
R2 = lambda zf: 1 - ((to7(zf[q] - z[q]) / M[q, None]) ** 2).sum() / SST
sub = lambda zf: answer(set_tape(H[q], zf[q]))
tapeof = lambda X: zc(run(X)[:, 9])
isM, isS = pos == st["pM"][:, None], pos == st["pS"][:, None]
rank = np.argsort(np.argsort(-x, 1), 1); pS0 = np.maximum(prevS, 0)

print(f"(a) {q.sum()} quiet final states.  accuracy of the unmodified network {np.mean(ans0[q] == st['pS'][q]):.3f}")
print("    set V                         kept | filler 0: R2  changed  accuracy | redraw the others of age >= 2: eta2  changed  accuracy | redraw all others: eta2  changed  accuracy | V unchanged")
sets = {"M, S": (isM | isS, np.broadcast_to(st["S"][:, None], x.shape)), "top 3": (rank < 3, np.broadcast_to(st["T3"][:, None], x.shape)),
        "top 4": (rank < 4, np.broadcast_to(np.sort(x, 1)[:, -4][:, None], x.shape)),
        "candidates (investigator)": ((x > prevS) | (x > 0.75 * pS0), np.where(prevS > 0, 0.75 * prevS, prevS))}
for f in (1.0, 0.9, 0.75, 0.6, 0.5, 0.0):
    sets[f"x > {f} max(S_prev, 0)"] = (x > f * pS0, f * pS0)
for n, (keep, thr) in sets.items():
    z0 = tapeof(np.where(keep, x, 0.0)); a = sub(z0); out = [f"{R2(z0):.3f}  {np.mean(a != ans0[q]):.3f}  {np.mean(a == st['pS'][q]):.3f}"]
    for only_old in (True, False):
        redraw = ~keep & (old if only_old else True)
        xr = np.where(redraw, ndtri(rng.random(x.shape) * ndtr(np.where(redraw, thr, 0.0))), x)
        if n.startswith("x >"): f_ = float(n.split()[2]); same = np.array_equal((xr > f_ * np.maximum(prev_top2(xr)[1], 0))[q], keep[q])
        else: same = "-"
        zr = tapeof(xr); a = sub(zr); r = R2(zr); out.append(f"{1 - (1 - r) / 2:.3f}  {np.mean(a != ans0[q]):.3f}  {np.mean(a == st['pS'][q]):.3f}")
    print(f"    {n:28s} {keep[q].sum(1).mean():5.2f} | " + " | ".join(out) + f" | {same}")
a = sub(from7(c.mean(0))[None] * M[:, None]); print(f"    mean tape: changed {np.mean(a != ans0[q]):.3f}, accuracy {np.mean(a == st['pS'][q]):.3f}")

print("\n(b) Unit 5's write threshold in random sequences.  Inputs x_p (p = 2..8) below the runner-up so far S_prev > 0: share for which unit 5 is on at step p + 1")
print("    (h_(p+1)[5] > 0.02 M_prev), by x_p / S_prev;  all sequences, no quiet filter")
P_, S_, M_, X_, U5 = [], [], [], [], []
for p in range(2, 9):
    s = (prevS[:, p] > 0) & (x[:, p] < prevS[:, p])
    S_.append(prevS[s, p]); M_.append(prevM[s, p]); X_.append(x[s, p]); U5.append(Hall[s, p + 1, 5]); P_.append(np.full(s.sum(), p))
S_, M_, X_, U5, P_ = map(np.concatenate, (S_, M_, X_, U5, P_)); fr = X_ / S_; on = U5 > 0.02 * M_
edges = [-np.inf, 0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
print("    x_p / S_prev:   " + " ".join(f"{b:>7}" for b in ("<0", "0-.3", ".3-.5", ".5-.6", ".6-.7", ".7-.75", ".75-.8", ".8-.9", ".9-1")))
for lab, s in (("all", np.ones(len(fr), bool)), ("S_prev/M_prev < 0.5", S_ / M_ < 0.5), ("S_prev/M_prev > 0.8", S_ / M_ > 0.8)):
    print(f"    {lab:20s} " + " ".join(f"{on[s & (fr >= a) & (fr < b)].mean():7.3f}" for a, b in zip(edges[:-1], edges[1:])) + f"   n = {s.sum()}")
print("    exact threshold by a sweep of x_p (grid 0.01 S_prev from -0.5 to 1) in 20000 random prefixes: Theta = first value at which unit 5 at p + 1 exceeds 0.02 M_prev")
n0 = 20000; F = np.arange(-0.5, 1.0, 0.01); th = []; cfg = []
for p in range(2, 9):
    s = np.flatnonzero(prevS[:n0, p] > 0)
    X = np.repeat(x[s, None, : p + 2], len(F), 1); X[:, :, p] = F * prevS[s, p, None]
    u5 = run(X.reshape(-1, p + 2))[:, p + 1, 5].reshape(len(s), len(F)) > 0.02 * prevM[s, p, None]
    th.append(np.where(u5.any(1), F[u5.argmax(1)], np.nan)); cfg.append(np.stack([prevS[s, p] / prevM[s, p], np.full(len(s), p)], 1))
th = np.concatenate(th); cfg = np.concatenate(cfg); qn = lambda v: f"median {np.nanmedian(v):.2f}, 10%..90% {np.nanquantile(v, .1):.2f}..{np.nanquantile(v, .9):.2f}"
print(f"    Theta / S_prev over {len(th)} prefixes: {qn(th)};  never on below S_prev: {np.isnan(th).mean():.3f};  already on at -0.5 S_prev: {np.mean(th == F[0]):.3f};  within 0.65..0.85: {np.nanmean((th > 0.65) & (th < 0.85)):.3f}")
for lab, s in (("S_prev/M_prev < 0.4", cfg[:, 0] < 0.4), ("0.4..0.7", (cfg[:, 0] >= 0.4) & (cfg[:, 0] < 0.7)), ("> 0.7", cfg[:, 0] >= 0.7)):
    print(f"      {lab:20s} {qn(th[s])}   n = {s.sum()}")

print("\n(c1) CLAIM 6: consecutive quiet states t -> t + 1, t = 5..8")
k_p = {u: LEFTM @ W[ORD, u] for u in PULSE}; rows = []
for t in range(5, 9):
    qq = top3(x, t)["quiet"] & top3(x, t + 1)["quiet"]; c0, c1 = zc(Hall[qq, t]), zc(Hall[qq, t + 1])
    lin = LAMV * c0 + x[qq, t + 1, None] * kw + sum(Hall[qq, t, u, None] * k_p[u] for u in PULSE)
    ncl = (Zall[qq, t + 1][:, ORD] < 0).sum(1); ncomp = (Zall[qq, t + 1][:, COMP] < 0).sum(1)
    rows.append(np.stack([nrm(c1 - LAMV * c0) / nrm(c1), nrm(c1 - lin) / nrm(c1), ncl, ncomp, top3(x, t)["aM"][qq]], 1))
R = np.concatenate(rows)
print(f"    {len(R)} pairs.  no non-pulse unit clipped {np.mean(R[:, 2] == 0):.3f}, one {np.mean(R[:, 2] == 1):.3f}, a comparator clipped {np.mean(R[:, 3] > 0):.3f}")
print(f"    |c(t+1) - lam c(t)| / |c(t+1)|: median {np.median(R[:, 0]):.3f}, 90% {np.quantile(R[:, 0], .9):.3f};  after subtracting direct and pulse writes: median {np.median(R[:, 1]):.3f}, 90% {np.quantile(R[:, 1], .9):.3f}")
print("    by age of the maximum (median of the second error): " + "  ".join(f"a_M={k}: {np.median(R[R[:, 4] == k, 1]):.3f}" for k in range(2, 9)))
print(f"    one unit clipped: median of the second error {np.median(R[R[:, 2] == 1, 1]):.3f}, 90% {np.quantile(R[R[:, 2] == 1, 1], .9):.3f}")

print("\n(c2) CLAIM 8: capacity of the dictionary c = sum_e A_e kA lam^(a_e - 2)")
kA = from7(np.load(HERE / "v1_patterns.npz")["kA"]); AG = np.arange(2, 10)
Phi = np.stack([to7(kA * LAMV ** (a - 2)) for a in AG], 1); sv = np.linalg.svd(Phi)
print("    singular values", np.round(sv[1], 3), f"condition number {sv[1][0] / sv[1][-1]:.1f};  null direction", np.round(np.abs(sv[2][-1]), 2))
for sigma in (0.0, 0.3, 1.0):
    out = []
    for N in (1, 2, 3, 4, 5):
        n = 20000; idx = np.argsort(rng.random((n, 8)), 1)[:, :N]; amp = np.zeros((n, 8)); np.put_along_axis(amp, idx, rng.uniform(1, 5, (n, N)), 1)
        cc = amp @ Phi.T + sigma * rng.standard_normal((n, 7)); best = np.zeros((n, N), int); br = np.full(n, np.inf)
        for sb in combinations(range(8), N):
            Pm = Phi[:, sb]; res = ((cc - cc @ np.linalg.pinv(Pm).T @ Pm.T) ** 2).sum(1); b = res < br - 1e-12; br[b] = res[b]; best[b] = sb
        sc = tape13(from7(cc)) @ UO.T; topN = np.sort(7 - np.argsort(-sc[:, :8], 1)[:, :N], 1)
        out.append(f"N={N}: {np.mean((np.sort(best, 1) == np.sort(idx, 1)).all(1)):.2f} / {np.mean((topN == np.sort(idx, 1)).all(1)):.2f}")
    print(f"    noise sd {sigma}: ages read back by subset least squares / by the top entries of the tape piece:  " + "   ".join(out))
Ah = c @ np.linalg.pinv(Phi).T; u = np.stack([Hall[q, 9 - a + 1, 5] for a in AG], 1) / M[q, None]
sc = H[q][:, ORD] @ P_TAPE.T @ UO.T
print("    real quiet final states: correlation of slot amplitude with unit 5's pulse of that age (ages 2..9): " + " ".join(f"{np.corrcoef(Ah[:, j], u[:, j])[0, 1]:.2f}" for j in range(8)))
print(f"    largest slot = runner-up's age {np.mean(AG[Ah.argmax(1)] == st['aS'][q]):.3f};  largest entry of the tape piece (ages >= 2) = runner-up {np.mean(sc[:, :8].argmax(1) == st['pS'][q]):.3f};"
      f"  full score {np.mean((H[q] @ U.T)[:, :8].argmax(1) == st['pS'][q]):.3f};  R2 of c = Phi u (uncentred) {1 - ((c - u @ Phi.T) ** 2).sum() / (c ** 2).sum():.3f}")
