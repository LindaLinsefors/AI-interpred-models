"""S4. Is there a better-defined tape?  And what is lost when one mode is deleted from the STATE?
A. Deleting one mode's part from a quiet state at step t0 and continuing the run with the real inputs: share of
   answers that change, accuracy, share of perturbed states with a negative activation, and the same with the
   perturbed state clipped at zero.
B. All 512 subsets of the nine modes as candidate "tapes" (these are all the subspaces that are closed under the
   resting map): P(runner-up on top among ages >= 2) of the subset's piece of the scores, quiet final states.
C. Other definitions: the part of the scores in the seven eigen-directions of the shift matrix
   Sigma = U_o A pinv(U_o) whose eigenvalues match the tape modes; the readout's own view (scores of the 13
   non-pulse units); and the best LINEAR decoder of the runner-up's position from the 7 tape coordinates alone
   (multinomial logistic regression without intercept), from the 6 other coordinates, and from both.
D. Persistence over one real step: is (entry - runner-up's entry) at step 9 equal to the same thing at step 8,
   for the tape piece and for the candidates?  (steps on which x_9 is below the runner-up, state at 8 quiet)
Run:  python -B s4_alternatives.py > s4_alternatives.txt"""
from itertools import combinations

from sklearn.linear_model import LogisticRegression

from t1_common import *

# ---------------------------------------------------------------- A. deletion from the state
B = 200_000
x = sample(B, 401); truth = second_argmax(x)
PROJ = {m: projector([m]) for m in NAMES}
def delete(h, names):
    out = h.copy(); out[:, ORD] -= h[:, ORD] @ sum(PROJ[m] for m in names).T
    return out
print(f"A. Deleting one mode's part from a quiet state at step t0, then continuing with the real inputs ({B} sequences, seed 401)")
print("   columns per t0: answers changed / accuracy after (accuracy before in the header) ; second line: share of perturbed states with a negative entry, and changed / accuracy with the state clipped at zero")
H = run(x)
for t0 in (5, 7, 8, 9):
    q = prefix_stats(x, t0)["quiet"]; h = H[q, t0]; fut = x[q, t0 + 1:]
    fin = lambda hh: answer(continue_from(hh, fut)[:, -1]) if t0 < 9 else answer(hh)
    a0 = fin(h); acc0 = np.mean(a0 == truth[q])
    print(f"   t0 = {t0}: {q.sum()} quiet states, accuracy before {acc0:.3f}")
    for nm, names in [(m, [m]) for m in NAMES] + [("TAPE (4)", TAPE), ("lvl+run+clk", ["level", "runner", "clock"])]:
        hp = delete(h, names); neg = (hp < -1e-9).any(1)
        a1 = fin(hp); a2 = fin(np.maximum(hp, 0))
        print(f"      {nm:12s} changed {np.mean(a1 != a0):.3f}  accuracy {np.mean(a1 == truth[q]):.3f}   | negative {neg.mean():.3f}  clipped: changed {np.mean(a2 != a0):.3f}  accuracy {np.mean(a2 == truth[q]):.3f}")

# ---------------------------------------------------------------- B. all subsets of modes
d = population(400_000, 402); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; top_full = top_entry(full, d["valid"])
print(f"\nB. All subsets of the nine modes ({n} quiet final states, 400000 sequences, seed 402).  Full score: {np.mean(top_full == d['iS']):.3f}")
res = {}
for k in range(1, 10):
    for sub in combinations(NAMES, k):
        res[sub] = np.mean(top_entry(sum(pc[m] for m in sub), d["valid"]) == d["iS"])
for k in range(1, 10):
    best = sorted(((v, s) for s, v in res.items() if len(s) == k), reverse=True)[:3]
    print(f"   size {k}: " + ";  ".join(f"{'+'.join(s)} {v:.3f}" for v, s in best))
four = sorted(((v, s) for s, v in res.items() if len(s) == 4), reverse=True)
rank = [s for _, s in four].index(tuple(m for m in NAMES if m in TAPE)) + 1
print(f"   the tape (h2+alt+h3+short) ranks {rank} of {len(four)} four-mode subsets; next best four: {'+'.join(four[1][1])} {four[1][0]:.3f}")
print("   every subset that reaches 0.96: " + ";  ".join(f"{'+'.join(s)} {v:.3f}" for s, v in res.items() if v >= 0.96))

# ---------------------------------------------------------------- C. other definitions
A = W[np.ix_(ORD, ORD)]
Sig = Uo @ A @ np.linalg.pinv(Uo); mu, E = np.linalg.eig(Sig); Einv = np.linalg.inv(E)
targets = [LAM["alt"], LAM["h2"], np.conj(LAM["h2"]), LAM["h3"], np.conj(LAM["h3"]), LAM["short"], np.conj(LAM["short"])]
sel = [int(np.argmin(np.abs(mu - t_))) for t_ in targets]
print("\nC. Eigenvalues of Sigma (modulus at angle): " + ", ".join(f"{abs(m_):.3f}@{np.degrees(np.angle(m_)):.0f}" for m_ in mu))
print("   chosen as tape-like: " + ", ".join(f"{abs(mu[k]):.3f}@{np.degrees(np.angle(mu[k])):.0f}" for k in sel))
Pi = np.real(E[:, sel] @ Einv[sel, :])
so = d["h"][:, ORD] @ Uo.T
tape = sum(pc[m] for m in TAPE)
cands = {"four-mode tape piece": tape, "Sigma-tape: 7 eigen-directions of Sigma": so @ Pi.T, "Sigma-rest: the 3 other directions": so - so @ Pi.T,
         "scores of the 13 non-pulse units (readout's view)": so, "tape + runner + clock + trans (7 modes)": tape + pc["runner"] + pc["clock"] + pc["trans"],
         "all modes but level (8 modes)": so - pc["level"], "full score": full}
print("   candidate                                              P(S top)  agree   slope  err_sd  err_iqr")
for nm, Y in cands.items():
    r = measures(Y, d, top_full)
    print(f"   {nm:54s} {r[0]:.3f}    {r[1]:.3f}  {r[2]:6.1f}  {r[3]:.4f}  {r[4]:.4f}")
print(f"   overlap: share of the Sigma-tape piece's variance (indices 0-7, centred) that is the four-mode tape piece: "
      f"{1 - ((so @ Pi.T - tape)[:, :8] ** 2).sum() / ((so @ Pi.T)[:, :8] ** 2).sum():.3f}")
# best linear decoders of the runner-up's position (8 classes = ages 2..9), no intercept, coordinates / M
c = coords(d["h"]); M = d["M"][:, None]
Xt = tape_coords(d["h"]) / M
Xr = np.stack([c["level"], c["runner"], c["clock"].real, c["clock"].imag, c["trans"], c["echo"]], 1) / M
tr = ar < 100_000; te = ar >= n // 2
print("   best linear decoder of the runner-up's position among ages >= 2 (multinomial logistic regression, no intercept, coordinates / M, fit on 100000 states, scored on the second half):")
for nm, X in (("the 7 tape coordinates", Xt), ("the 6 other coordinates (level, runner, clock, trans, echo)", Xr), ("all 13 coordinates", np.concatenate([Xt, Xr], 1)),
              ("tape + clock (9)", np.concatenate([Xt, Xr[:, 2:4]], 1)), ("tape + level + runner (9)", np.concatenate([Xt, Xr[:, :2]], 1)),
              ("tape + level + runner + clock (11)", np.concatenate([Xt, Xr[:, :4]], 1))):
    sd = X[tr].std(0)
    clf = LogisticRegression(C=100.0, fit_intercept=False, max_iter=400).fit(X[tr] / sd, d["iS"][tr])
    print(f"      {nm:62s} accuracy {clf.score(X[te] / sd, d['iS'][te]):.3f}")
print(f"      for comparison, the network's own readout of the same coordinates: tape piece {np.mean(top_entry(tape, d['valid']) == d['iS']):.3f}, all 13 {np.mean(top_entry(so, d['valid']) == d['iS']):.3f}")

# ---------------------------------------------------------------- D. persistence over a real step
B = 400_000
x = sample(B, 403); st = prefix_stats(x, 8)
q = st["quiet"] & (x[:, 9] < st["S"]); xq = x[q]; nq = q.sum(); arq = np.arange(nq)
x8 = np.zeros((nq, 10)); x8[:, 1:] = xq[:, :9]
h8, h9 = run(x8)[:, 9], run(xq)[:, 9]
aS, aM, Mq = st["aS"][q], st["aM"][q], st["M"][q]
def cand(h):
    p = pieces(h); so_ = h[:, ORD] @ Uo.T; tp = sum(p[m] for m in TAPE)
    return {"four-mode tape piece": tp, "Sigma-tape": so_ @ Pi.T, "tape + runner + clock + trans": tp + p["runner"] + p["clock"] + p["trans"],
            "all modes but level": so_ - p["level"], "full score": h @ U.T, "rest piece (full - tape)": h @ U.T - tp}
c8, c9 = cand(h8), cand(h9)
print(f"\nD. Persistence over the step 8 -> 9 ({nq} sequences: state at 8 quiet and x_9 below the runner-up; seed 403).")
print("   D_8(a) = (entry of age a - runner-up's entry)/M at step 8, D_9(a + 1) the same entry one step later; entries of age 2..8 at step 8, not the runner-up")
print("   candidate                          below the runner-up: corr, slope of D_9 on D_8, rms(D_9 - D_8), rms D_8   |  the maximum's entry: corr, rms(D_9 - D_8), rms D_8")
a8 = np.arange(2, 9)[None, :]                                     # ages at step 8 -> index 9 - a at step 8, 8 - a at step 9
vals = x8[:, 9 - np.arange(2, 9)]; below = (vals < st["S"][q][:, None]) & (a8 != aS[:, None])
for nm in c8:
    D8 = (c8[nm][:, 9 - np.arange(2, 9)] - c8[nm][arq, 9 - aS][:, None]) / Mq[:, None]
    D9 = (c9[nm][:, 8 - np.arange(2, 9)] - c9[nm][arq, 8 - aS][:, None]) / Mq[:, None]
    u, v_ = D8[below], D9[below]; ism = a8 == aM[:, None]; um, vm = D8[ism], D9[ism]
    print(f"   {nm:32s}   {np.corrcoef(u, v_)[0, 1]:.3f}  {np.polyfit(u, v_, 1)[0]:.3f}  {np.sqrt(((v_ - u) ** 2).mean()):6.2f}  {np.sqrt((u ** 2).mean()):6.2f}"
          f"                      |  {np.corrcoef(um, vm)[0, 1]:.3f}  {np.sqrt(((vm - um) ** 2).mean()):6.2f}  {np.sqrt((um ** 2).mean()):6.2f}")
print("   the same for entries below the runner-up, split by the age of the entry at step 8 (rms(D_9 - D_8) / rms D_8):")
for nm in ("four-mode tape piece", "full score"):
    D8 = (c8[nm][:, 9 - np.arange(2, 9)] - c8[nm][arq, 9 - aS][:, None]) / Mq[:, None]
    D9 = (c9[nm][:, 8 - np.arange(2, 9)] - c9[nm][arq, 8 - aS][:, None]) / Mq[:, None]
    print(f"      {nm:22s} " + "  ".join(f"a={a}: {np.sqrt(((D9 - D8)[:, j][below[:, j]] ** 2).mean()) / np.sqrt((D8[:, j][below[:, j]] ** 2).mean()):.2f}" for j, a in enumerate(range(2, 9))))
