"""N2 check of sections 4 and 5: sampled quantities (N(0,1), fresh seeds 9201-9202, 2 x 300k sequences)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from modes import LEFT, coords

w4 = w[4]; g = -w[COMP]
x = np.concatenate([sample(300_000, 9201), sample(300_000, 9202)])
B = len(x)
H, Z = run(x, return_pre=True)
G = Z > 0
st = running_stats(x)
M, S, a1, a2 = st["M1"], st["M2"], st["a1"], st["a2"]
Mp = np.full_like(M, -np.inf); Mp[:, 1:] = M[:, :-1]            # max of x_0..x_{t-1}
Sp = np.full_like(S, -np.inf); Sp[:, 1:] = S[:, :-1]            # runner-up of x_0..x_{t-1}
a1p = np.full_like(a1, -1); a1p[:, 1:] = a1[:, :-1] + 1         # age of the previous max at step t (if not overtaken)
tt = np.broadcast_to(np.arange(10), x.shape)
newmax = (x > Mp) & (tt >= 1); newru = (x > Sp) & (x < Mp) & (tt >= 1); below = (x < Sp) & (tt >= 1)
print("accuracy", (logits(H).argmax(1) == second_argmax(x)).mean())

print("\n== section 5 table: shares of steps 1-9 ==")
n19 = B * 9
u5next = np.zeros_like(G[:, :, 0]); u5next[:, :-1] = G[:, 1:, 5]
print("new max %.4f  new runner-up %.4f  below runner-up %.4f" % (newmax.sum() / n19, newru.sum() / n19, below.sum() / n19))
b8 = below & (tt <= 8)
print("below runner-up, t<=8: unit 5 on at t+1: %.4f   (of those with S_prev>0: %.4f)" % (u5next[b8].mean(), u5next[b8 & (Sp > 0)].mean()))
print("  -> 'between Theta5 and S' share of steps 1-9 if rate applied to all below-steps: %.4f ; counted on t<=8 only: %.4f" % (u5next[b8].mean() * below.sum() / n19, (u5next & b8).sum() / n19))
posmax = (x > np.maximum(Mp, 0)); posmax[:, 0] = x[:, 0] > 0
print("unit 4 on | positive new max (t>=0): %.5f ; | negative new max (t>=1): %.6f ; share of unit-4 firings that are not positive new maxima: %.4f" % (G[..., 4][posmax].mean(), G[..., 4][newmax & (x < 0)].mean(), 1 - posmax[G[..., 4]].mean()))
pm8 = posmax & (tt <= 8); pr8 = (x > np.maximum(Sp, 0)) & (x < Mp) & (tt >= 1) & (tt <= 8)
print("unit 5 on at t+1 | positive new max %.4f ; | positive new runner-up %.4f ; | positive new runner-up below theta4 (unit 4 off) %.4f" % (u5next[pm8].mean(), u5next[pr8].mean(), u5next[pr8 & ~G[..., 4]].mean()))
r = x / Sp
sel = below & (tt >= 2) & (tt <= 8) & (Sp > 0) & ~G[..., 4]
for lo, hi in ((-np.inf, 0), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
    k = sel & (r >= lo) & (r < hi); print("  unit 5 on after x/S in [%.1f, %.1f): %.4f" % (lo, hi, u5next[k].mean()))

print("\n== 4.1 property 1: d theta_j(t+1) / d x_t with all four comparators on ==")
allon = G[..., COMP].all(-1) & (tt <= 8)
D = ((G * w) @ W[COMP].T) / g                                    # (B, T, 4): derivative through all 16 units
for nm, k in (("all four on", allon), ("all four on, unit 4 off", allon & ~G[..., 4])):
    d = np.abs(D[k]); print(f"  {nm}: n={k.sum()}  mean {np.round(d.mean(0), 4)}  99% {np.round(np.quantile(d, 0.99, axis=0), 3)}  max {np.round(d.max(0), 3)}")
print("  share of all steps with all four on: %.3f (t=0 %.3f, t=9 %.3f); none on %.3f; 1-3 on %.3f" % (G[..., COMP].all(-1).mean(), G[:, 0][:, COMP].all(-1).mean(), G[:, 9][:, COMP].all(-1).mean(), (~G[..., COMP].any(-1)).mean(), 1 - G[..., COMP].all(-1).mean() - (~G[..., COMP].any(-1)).mean()))

print("\n== 3.4 / 5.4: all four comparators cut at a new max over a positive old max ==")
nm = newmax & (Mp > 0); cut4 = ~G[..., COMP].any(-1)
print("  pooled s=1..9 %.4f ; s=1..8 %.4f ; s=2..6 %.4f" % (cut4[nm].mean(), cut4[nm & (tt <= 8)].mean(), cut4[nm & (tt >= 2) & (tt <= 6)].mean()))
print("  by step s=1..9:", np.round([cut4[nm & (tt == s)].mean() for s in range(1, 10)], 3))
ex = (x - Mp) / Mp
print("  excess <= 5%% of old max, by step s=1..9: %s ; pooled %.3f" % (np.round([cut4[nm & (tt == s) & (ex <= 0.05)].mean() for s in range(1, 10)], 2), cut4[nm & (ex <= 0.05)].mean()))
print("  excess > 30%%: by step %s ; pooled %.4f" % (np.round([cut4[nm & (tt == s) & (ex > 0.3)].mean() for s in range(1, 10)], 3), cut4[nm & (ex > 0.3)].mean()))
print("  unit 9 on at s+1 by s=1..8:", np.round([G[:, s + 1, 9][nm[:, s]].mean() for s in range(1, 9)], 4))
nxt = np.zeros_like(newmax); nxt[:, :-1] = newmax[:, 1:]
k = nm & (tt >= 1) & (tt <= 7); off0 = np.zeros_like(newmax); off0[:, :-2] = ~G[:, 2:, 0]
print("  unit 0 off at s+2 (s=1..7): all %.4f ; next input not a new max %.4f ; next input a new max %.4f ; P(next is new max) %.4f" % (off0[k].mean(), off0[k & ~nxt].mean(), off0[k & nxt].mean(), nxt[k].mean()))

print("\n== 5.3: comparators cut by a positive new runner-up ==")
pr = (x > np.maximum(Sp, 0)) & (x < Mp) & (tt >= 1)
ncut = (~G[..., COMP]).sum(-1)
print("  n cut = 0,1,2,3,4:", np.round([(ncut[pr] == k).mean() for k in range(5)], 4))
print("  none cut by age of max at arrival 1..8:", np.round([(ncut[pr & (a1p == a)] == 0).mean() for a in range(1, 9)], 3))
th = (H[:, :-1] @ W[COMP].T) / g                                  # thresholds met by x_{t+1}: (B, 9, 4)
th4 = -(H[:, :-1] @ W[4]) / w4
low = (th.min(-1) - np.maximum(Sp[:, 1:], 0)) / Mp[:, 1:]
k = pr[:, 1:] & (Sp[:, 1:] > 0)
print("  median (lowest threshold - S)/M at a positive new runner-up, ages 5..8:", np.round([np.median(low[k & (a1p[:, 1:] == a)]) for a in range(5, 9)], 3))

print("\n== 4.1 property 4: theta_4 / old max, by age ==")
k0 = (Mp[:, 1:] > 0.3)
print("  age of the max at the step where theta_4 is met (a1 of previous state + 1) = 1..9:", np.round([np.mean((th4 / Mp[:, 1:])[k0 & (a1p[:, 1:] == a)]) for a in range(1, 10)], 3))
kn = k0 & newmax[:, 1:]
print("  same, only steps where a new max arrives:", np.round([np.mean((th4 / Mp[:, 1:])[kn & (a1p[:, 1:] == a)]) for a in range(1, 10)], 3))

print("\n== 4.1 laws ==")
c = coords(H)
k = (tt >= 1)
Mprev0 = np.maximum(Mp, 0)
def r2c(y, yhat): return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
y = c["level"][k] / w4; xx = Mprev0[k]; b = (xx @ y) / (xx @ xx); print("  level/w4 = %.4f max(M_{t-1},0), centred R2 %.4f" % (b, r2c(y, b * xx)))
y = H[..., 2][k]; b = (xx @ y) / (xx @ xx); print("  h2 = %.3f max(M_{t-1},0), centred R2 %.4f" % (b, r2c(y, b * xx)))
Sp0 = np.maximum(S, 0)
for nm_, k in (("all states with M>0, t>=1", (M > 0) & (tt >= 1)), ("a1>=2, S>0.05", (a1 >= 2) & (S > 0.05)), ("a1>=2, a2>=2, S>0.05", (a1 >= 2) & (a2 >= 2) & (S > 0.05))):
    X = np.stack([M[k], Sp0[k]], 1); y = c["runner"][k] / w4; co = np.linalg.lstsq(X, y, rcond=None)[0]
    print("  runner/w4 = %.2f M + %.2f S+  (%s; centred R2 %.3f)" % (co[0], co[1], nm_, r2c(y, X @ co)))
print("  by age of max 1..9 (S>0.05): " + "; ".join("%.2fM+%.2fS" % tuple(np.linalg.lstsq(np.stack([M[k], Sp0[k]], 1), c["runner"][k] / w4, rcond=None)[0]) for k in [(a1 == a) & (S > 0.05) for a in range(1, 10)]))
rest = G[..., ORD].all(-1) & ~G[..., PULSE].any(-1)
print("  resting pattern: all %.4f, t=9 %.4f ; by a1=0..9 (S>0.05): %s" % (rest.mean(), rest[:, 9].mean(), np.round([rest[(a1 == a) & (S > 0.05)].mean() for a in range(10)], 4)))
