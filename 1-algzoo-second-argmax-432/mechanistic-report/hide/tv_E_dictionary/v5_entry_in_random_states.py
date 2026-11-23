"""V5. Does the "clean dictionary entry" (claim 1) hold in RANDOM quiet final states, not only in controlled sequences?
Cases: a quiet final state (t = 9) and an input x_p with max(p_M, p_S) < p <= 7 (it arrived after both, age a = 9 - p >= 2) and
0.5 S < x_p < S.  Its write is D = tape(x) - tape(x with x_p set to 0), in units of M.  Predictions:
  P1  the real extra pulse of unit 5 one step later, du = h_(p+1)[5](x) - h_(p+1)[5](x_p = 0):   du k5 lam^(a-2)
  P2  P1 + the direct write (l.w[o]) x_p lam^a
  P3  the investigator's formula  19.1 (x_p - 0.75 S)_+ kA lam^(a-2)   (kA = measured age-2 pattern, alt = 1)
err = |D - P| / |D| per case;  R2u = 1 - sum |D - P|^2 / sum |D|^2 (uncentred).  300000 iid N(0,1) sequences, seed 7171.
Run:  python -B v5_entry_in_random_states.py > v5_entry_in_random_states.txt"""
from v_common import *

x = sample(300_000, 7171); Hall = run(x); st = top3(x, 9); q = st["quiet"]; M, S = st["M"], st["S"]
k5, kw = LEFTM @ W[ORD, 5], LEFTM @ w[ORD]; kA = from7(np.load(HERE / "v1_patterns.npz")["kA"])
rows = []
for p in range(2, 8):
    s = q & (np.maximum(st["pM"], st["pS"]) < p) & (x[:, p] > 0.5 * S) & (x[:, p] < S)
    x0 = x[s].copy(); x0[:, p] = 0.0; H0 = run(x0); a = 9 - p
    D = (zc(Hall[s, 9]) - zc(H0[:, 9])) / M[s, None]; du = (Hall[s, p + 1, 5] - H0[:, p + 1, 5]) / M[s]
    P1 = du[:, None] * k5 * LAMV ** (a - 2); P2 = P1 + (x[s, p] / M[s])[:, None] * kw * LAMV ** a      # the pulse at p + 1 is written at p + 2
    P3 = (19.1 * np.maximum(x[s, p] - 0.75 * S[s], 0) / M[s])[:, None] * kA * LAMV ** (a - 2)
    later = (x[s, p + 1:] > 0.5 * S[s, None]).any(1)                       # another large input arrives after it
    rows.append(np.column_stack([np.full(s.sum(), a), x[s, p] / S[s], du, later, nrm(D), nrm(D - P1), nrm(D - P2), nrm(D - P3)]))
R = np.concatenate(rows)
print(f"{len(R)} cases in {q.sum()} quiet final states.  unit 5 fired (du > 0.02) in {np.mean(R[:, 2] > 0.02):.3f} of them; by x_p / S: " +
      "  ".join(f"[{a_:.2f},{b_:.2f}) {np.mean(R[(R[:, 1] >= a_) & (R[:, 1] < b_), 2] > 0.02):.2f}" for a_, b_ in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.9), (0.9, 1.0))))


def rep(lab, s):
    o = []
    for j, n in ((5, "P1 real pulse"), (6, "P2 + direct write"), (7, "P3 formula")):
        e = R[s, j] / np.maximum(R[s, 4], 1e-12)
        o.append(f"{n}: R2u {1 - (R[s, j] ** 2).sum() / (R[s, 4] ** 2).sum():6.3f}, err median {np.median(e):.3f}, 90% {np.quantile(e, .9):.3f}, err < 0.2 in {np.mean(e < 0.2):.2f}")
    print(f"   {lab:44s} n = {s.sum():6d}  rms |D| {np.sqrt((R[s, 4] ** 2).mean()):5.2f} | " + " | ".join(o))


fired = R[:, 2] > 0.02
rep("all cases", np.ones(len(R), bool)); rep("unit 5 fired", fired); rep("unit 5 did not fire", ~fired)
rep("fired, no later input above 0.5 S", fired & (R[:, 3] == 0)); rep("fired, a later input above 0.5 S", fired & (R[:, 3] == 1))
for a in range(2, 8):
    rep(f"fired, age {a}", fired & (R[:, 0] == a))
