"""N2: a few follow-up checks (N(0,1), seeds 9211-9212, 2 x 300k sequences)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import LEFT, coords

w4 = w[4]; g = -w[COMP]
x = np.concatenate([sample(300_000, 9211), sample(300_000, 9212)])
H, Z = run(x, return_pre=True); G = Z > 0
st = running_stats(x); M, S, a1, a2, p1, p2 = st["M1"], st["M2"], st["a1"], st["a2"], st["p1"], st["p2"]
tt = np.broadcast_to(np.arange(10), x.shape)
Mp = np.full_like(M, -np.inf); Mp[:, 1:] = M[:, :-1]
Sp = np.full_like(S, -np.inf); Sp[:, 1:] = S[:, :-1]
a1p = np.full_like(a1, -1); a1p[:, 1:] = a1[:, :-1] + 1
newmax = (x > Mp) & (tt >= 1)
cut4 = ~G[..., COMP].any(-1)

print("== V2's selection for the small-excess rate: new max at t=3..9, old max > 0.3 and at least 2 steps old on arrival ==")
sel = newmax & (Mp > 0.3) & (a1p >= 2) & (tt >= 3)
ex = np.where(sel, (x - Mp) / np.where(Mp > 0, Mp, 1), np.nan)
print("  all four cut, by t=3..9:", np.round([cut4[sel & (tt == t)].mean() for t in range(3, 10)], 3))
print("  excess <= 5%, by t=3..9:", np.round([cut4[sel & (tt == t) & (ex <= 0.05)].mean() for t in range(3, 10)], 2), " by age of old max 2..8:", np.round([cut4[sel & (a1p == a) & (ex <= 0.05)].mean() for a in range(2, 9)], 2))
sel0 = newmax & (Mp > 0)
print("  all new maxima over a positive old max, excess <= 5%: by age of old max on arrival 1..8:", np.round([cut4[sel0 & (a1p == a) & ((x - Mp) <= 0.05 * Mp)].mean() for a in range(1, 9)], 2))

print("\n== level/(w4 M) by age of the max (medians), and per-step growth ==")
lev = coords(H)["level"] / w4
k = (M > 0.3)
print("  median level/(w4 M) by a1=1..9:", np.round([np.median((lev / M)[k & (a1 == a)]) for a in range(1, 10)], 3))
r = lev[:, 1:] / lev[:, :-1]
quiet = (a1[:, :-1] >= 2) & (M[:, :-1] > 0.3) & ~newmax[:, 1:]
p = ~G[:, 1:, 5] & ~G[:, 1:, 9]
print("  per-step ratio level(t+1)/level(t), max age>=2, no new max: median %.4f, mean %.4f ; steps with units 5, 9 off at t+1: median %.4f mean %.4f (share %.2f) ; with unit 5 on: median %.4f mean %.4f" % (np.median(r[quiet]), r[quiet].mean(), np.median(r[quiet & p]), r[quiet & p].mean(), p[quiet].mean(), np.median(r[quiet & G[:, 1:, 5]]), r[quiet & G[:, 1:, 5]].mean()))
imp = run(np.eye(10)[0])[0]
print("  impulse (1,0,..,0): level/w4 at t=0..9:", np.round(coords(imp)["level"] / w4, 3))

print("\n== 5.1: the slot of an unrecorded low input one step later (score minus runner-up's score, units of M) ==")
s_all = H @ U.T                                             # (B, T, 10): provisional scores
for lab, cond in (("x_t < S/2 (S>0), unit 5 silent at t+1", lambda t: (x[:, t] < Sp[:, t] / 2)), ("x_t < S (S>0), unit 5 silent at t+1", lambda t: (x[:, t] < Sp[:, t]))):
    vals = []
    for t in range(2, 9):
        ok = cond(t) & (Sp[:, t] > 0) & ~G[:, t + 1, 5] & (x[:, t + 1] < Sp[:, t]) & (p2[:, t + 1] < t)
        idx_ru = 9 - (t + 1 - p2[ok, t + 1])
        d = (s_all[ok, t + 1, 8] - s_all[ok, t + 1, idx_ru]) / M[ok, t + 1]
        vals.append(d)
    v = np.concatenate(vals)
    print(f"  {lab}: median {np.median(v):.1f}, quartiles {np.quantile(v, .25):.1f} / {np.quantile(v, .75):.1f}, 10-90% {np.quantile(v, .1):.1f} / {np.quantile(v, .9):.1f}")
    print("     median by age of the max at t+1 = 1..8:", np.round([np.median(np.concatenate([((s_all[:, t + 1, 8] - s_all[np.arange(len(x)), t + 1, np.clip(9 - (t + 1 - p2[:, t + 1]), 0, 9)]) / M[:, t + 1])[cond(t) & (Sp[:, t] > 0) & ~G[:, t + 1, 5] & (x[:, t + 1] < Sp[:, t]) & (p2[:, t + 1] < t) & (a1[:, t + 1] == a)] for t in range(2, 9)])) for a in range(1, 9)], 1))

print("\n== unit 5 one step after a positive new max, by step s=0..8 ==")
posmax = (x > np.maximum(Mp, 0)); posmax[:, 0] = x[:, 0] > 0
print("  ", np.round([G[:, s + 1, 5][posmax[:, s]].mean() for s in range(9)], 4), " pooled s>=1: %.4f" % np.mean(np.concatenate([G[:, s + 1, 5][posmax[:, s]] for s in range(1, 9)])))

print("\n== lowest comparator threshold above S+, in units of M, all states with M>0.3, by age of the max when the threshold is met ==")
th = (H[:, :-1] @ W[COMP].T) / g
low = (th.min(-1) - np.maximum(S[:, :-1], 0)) / M[:, :-1]
kk = (M[:, :-1] > 0.3) & (tt[:, :-1] >= 2)
print("  median by age when met 2..9:", np.round([np.median(low[kk & (a1[:, :-1] + 1 == a)]) for a in range(2, 10)], 3))
