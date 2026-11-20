"""B3b. Which inputs make units 5 and 9 fire one step later (and so change the level)?
Class of x_u relative to the statistics before it (max standing and positive):
  low    : x_u <= S+                (not a new runner-up)
  runner : S+ < x_u, unit 4 silent  (new runner-up)
  false  : unit 4 fired although x_u <= M+
  record : x_u > M+
Data: N(0,1), seed 1035, 400k.
"""
from blib import *

x = sample(400_000, 1035)
H, Z = run(x, return_pre=True)
st = prefix_stats(x)
n = len(x)
Mp = pos(np.where(np.isfinite(st["M1prev"]), st["M1prev"], 0))        # before x_u
Sp = pos(np.where(np.isfinite(st["M2prev"]), st["M2prev"], 0))
cls = np.where(x <= Sp, 0, np.where(H[:, :, 4] > 0, 2, 1))
cls = np.where(x > Mp, 3, cls)
names = ["low", "runner", "false", "record"]
h5n = np.concatenate([H[:, 1:, 5], np.zeros((n, 1))], 1)              # h5 at u+1
h9n = np.concatenate([H[:, 1:, 9], np.zeros((n, 1))], 1)
L = LEFT["level"].real
k5, k9 = L @ W[ORD, 5], L @ W[ORD, 9]
ok = (np.arange(10)[None, :] <= 8) & (Mp > 0)                         # the max before x_u is positive, u+1 exists
print("[MEASURED n = 400k, seed 1035] steps u = 1..8 with a positive standing max before x_u")
print("class of x_u | share | P(h5(u+1) > 0) | mean h5(u+1)/(10.157 M) | P(h9(u+1) > 0) | mean h9(u+1)/(10.157 M) | level change at u+2 from units 5, 9: mean (k5 h5 + k9 h9)/(10.157 M)")
Mnow = np.maximum(Mp, x)
for c in range(4):
    s = ok & (cls == c)
    print(f"{names[c]} | {s.sum()/ok.sum():.4f} | {(h5n[s] > 0).mean():.4f} | {(h5n[s]/(W4*Mnow[s])).mean():.4f} | {(h9n[s] > 0).mean():.4f} | {(h9n[s]/(W4*Mnow[s])).mean():.4f} | "
          f"{((k5*h5n[s] + k9*h9n[s])/(W4*Mnow[s])).mean():+.4f}")
s = ok & (cls == 1)
y = h5n[s] / W4
A_ = np.column_stack([x[s], Sp[s], Mp[s]])
b, *_ = np.linalg.lstsq(A_, y, rcond=None)
r2 = 1 - ((y - A_ @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"new runner-up steps: h5(u+1)/10.157 = {b[0]:+.3f} x_u {b[1]:+.3f} S+_old {b[2]:+.3f} M   (R2c {r2:.3f}, n = {s.sum()})")
