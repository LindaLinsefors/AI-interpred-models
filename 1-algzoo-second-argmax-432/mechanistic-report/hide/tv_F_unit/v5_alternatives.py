"""V5. Claim 9 (without the decoders, see v5b): all 511 subsets of the nine modes, the Sigma-tape, persistence.
Own code and seeds.  Run:  python -B v5_alternatives.py > v5_alternatives.txt"""
from itertools import combinations

from vcommon import *

d = population(sample(400_000, 9501)); n = len(d["h"]); ar = np.arange(n)
pc = pieces(d["h"]); full = d["h"] @ U.T; topf = top_old(full, d["valid"]); tape = sum(pc[m] for m in TAPE)
pS = lambda Sc: np.mean(top_old(Sc, d["valid"]) == d["iS"])
print(f"{n} quiet final states (400000 sequences, seed 9501); full score {pS(full):.3f}")
res = {sub: pS(sum(pc[m] for m in sub)) for k in range(1, 10) for sub in combinations(MODES, k)}
for k in range(1, 10):
    best = sorted(((v, s) for s, v in res.items() if len(s) == k), reverse=True)[:3]
    print(f"   size {k}: " + ";  ".join(f"{'+'.join(s)} {v:.3f}" for v, s in best))
four = sorted(((v, s) for s, v in res.items() if len(s) == 4), reverse=True)
print(f"   the tape ranks {[set(s) for _, s in four].index(set(TAPE)) + 1} of {len(four)} four-mode subsets ({res[tuple(m for m in MODES if m in TAPE)]:.3f}); next {'+'.join(four[1][1])} {four[1][0]:.3f}")
print("   subsets reaching 0.96: " + ";  ".join(f"{'+'.join(s)} {v:.3f}" for s, v in res.items() if v >= 0.96))

Sig = Uo @ A @ np.linalg.pinv(Uo); mu, E = np.linalg.eig(Sig); Ei = np.linalg.inv(E)
targets = [EIG[m] for m in TAPE] + [np.conj(EIG[m]) for m in ("short", "h2", "h3")]
sel = sorted({int(np.argmin(np.abs(mu - t_))) for t_ in targets})
print("\nSigma = U_o A pinv(U_o): eigenvalues " + ", ".join(f"{abs(m_):.3f}@{np.degrees(np.angle(m_)):.0f}" for m_ in mu))
print("   taken as tape-like (" + str(len(sel)) + "): " + ", ".join(f"{abs(mu[k]):.3f}@{np.degrees(np.angle(mu[k])):.0f}" for k in sel))
Pi = np.real(E[:, sel] @ Ei[sel, :]); so = d["h"][:, ORD] @ Uo.T; sig = so @ Pi.T
for nm, Sc in (("four-mode tape piece", tape), ("Sigma-tape", sig), ("scores of the 13 non-pulse units", so), ("full", full)):
    b, sd, iq, _ = near_slope(Sc, d)
    print(f"   {nm:34s} P(S top) {pS(Sc):.3f}  agree {np.mean(top_old(Sc, d['valid']) == topf):.3f}  slope {b:6.1f}  err sd {sd:.4f}  half-IQR {iq:.4f}")
print(f"   share of the Sigma-tape's power (indices 0-7) that is the four-mode tape piece: {1 - ((sig - tape)[:, :8] ** 2).sum() / (sig[:, :8] ** 2).sum():.3f}")

# ---- persistence over the step 8 -> 9
x = sample(400_000, 9502); st = stats(x, 8); q = st["quiet"] & (x[:, 9] < st["S"]); H = run(x[q]); h8, h9 = H[:, 8], H[:, 9]
nq = q.sum(); arq = np.arange(nq); aS, aM, Mq, Sq = st["aS"][q], st["aM"][q], st["M"][q], st["S"][q]
ages = np.arange(2, 9)                                   # ages at step 8; index 8 - a at step 8 (scores of h_8 read with U: age a <-> index 9 - a)
vals = x[q][:, 8 - ages]; below = (vals < Sq[:, None]) & (ages[None, :] != aS[:, None]); ism = ages[None, :] == aM[:, None]
def cand(h):
    p = pieces(h); tp = sum(p[m] for m in TAPE); so_ = h[:, ORD] @ Uo.T
    return {"four-mode tape piece": tp, "full score": h @ U.T, "Sigma-tape": so_ @ Pi.T, "rest piece": h @ U.T - tp}
c8, c9 = cand(h8), cand(h9)
print(f"\nPersistence 8 -> 9 ({nq} sequences, state at 8 quiet, x_9 below the runner-up; seed 9502): entries below the runner-up: corr, rms change, rms size | the maximum's entry: corr, rms change, rms size")
for nm in c8:
    D8 = (c8[nm][:, 9 - ages] - c8[nm][arq, 9 - aS][:, None]) / Mq[:, None]          # at step 8 the entry of age a is index 9 - a
    D9 = (c9[nm][:, 8 - ages] - c9[nm][arq, 8 - aS][:, None]) / Mq[:, None]          # one step later it has age a + 1 = index 8 - a
    u, v_ = D8[below], D9[below]; um, vm = D8[ism], D9[ism]
    print(f"   {nm:22s} {np.corrcoef(u, v_)[0, 1]:.3f}  {np.sqrt(((v_ - u) ** 2).mean()):5.2f}  {np.sqrt((u ** 2).mean()):5.2f}   |  {np.corrcoef(um, vm)[0, 1]:.3f}  {np.sqrt(((vm - um) ** 2).mean()):5.2f}  {np.sqrt((um ** 2).mean()):5.2f}")
print("   attack: the same split by whether the step 8 -> 9 is a RESTING step for the 13 non-pulse units (all gates open at 9, pulse units silent at 8)")
Z9 = h8 @ W.T + x[q][:, 9:10] * w
resting = (Z9[:, ORD] > 0).all(1) & (h8[:, PULSE] == 0).all(1)
print(f"      share of these steps that are resting: {resting.mean():.3f}")
for nm in ("four-mode tape piece", "full score"):
    D8 = (c8[nm][:, 9 - ages] - c8[nm][arq, 9 - aS][:, None]) / Mq[:, None]; D9 = (c9[nm][:, 8 - ages] - c9[nm][arq, 8 - aS][:, None]) / Mq[:, None]
    for lab, r in (("resting", resting), ("not resting", ~resting)):
        b_ = below & r[:, None]
        print(f"      {nm:22s} {lab:12s} corr {np.corrcoef(D8[b_], D9[b_])[0, 1]:.3f}  rms change {np.sqrt(((D9 - D8)[b_] ** 2).mean()):5.2f}  rms size {np.sqrt((D8[b_] ** 2).mean()):5.2f}")
