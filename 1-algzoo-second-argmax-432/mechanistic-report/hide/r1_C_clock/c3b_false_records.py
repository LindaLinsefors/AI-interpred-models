"""C3b. New-runner-up inputs that DO fire unit 4 ('false records': S+ < x_t < M but x_t > theta_4): what happens to the clock?
Also: which events leave no immediate trace.  400k N(0,1) sequences, seed 1301."""
from cl import *
x, H, Z, st = gen(400_000, 1301)
tm = clock_terms(x, H, Z)
z = tm["z"] / K4
M, S, a1 = st["M1"], pos(st["M2"]), st["a1"]
G = Z > 0
th = thresholds(H)
bb, tt = np.nonzero((x[:, 1:9] > S[:, :8]) & (x[:, 1:9] < M[:, :8]) & (M[:, :8] > 0)); tt = tt + 1      # events at t = 1..8 (so that t+1 exists)
v, Mo, So, age = x[bb, tt], M[bb, tt - 1], S[bb, tt - 1], a1[bb, tt]
f4 = G[bb, tt, 4]
ncl = (~G[bb, tt][:, COMP]).sum(1)
print(f"events (S+ < x_t < M, t = 1..8): {bb.size};  unit 4 fires in {f4.mean():.4f};  nothing at all happens at the event step (no comparator clipped, unit 4 silent) in {np.mean((ncl == 0) & ~f4):.4f}")
print("by age of the max at the event:  share with unit 4 firing | share with nothing happening | among unit-4 firings: mean number of comparators clipped, share with all four clipped")
for a in range(1, 9):
    q = age == a
    print(f"   a={a}: {f4[q].mean():.3f} | {np.mean((ncl[q] == 0) & ~f4[q]):.3f} | {ncl[q & f4].mean() if (q & f4).any() else float('nan'):.2f}  {np.mean(ncl[q & f4] == 4) if (q & f4).any() else float('nan'):.3f}   n={q.sum()}")
print("\nfalse records (unit 4 fires): kick h4/K4 = v - theta_4;  position of theta_4, (theta_4 - S)/(M - S): median "
      f"{np.median(((th[bb, tt - 1, 4] - So) / (Mo - So))[f4]):.3f};  h4/K4 relative to (v - S_old): median {np.median((H[bb, tt, 4] / K4 / (v - So))[f4]):.3f}")
print("[FIT] next-step clock after a false record:  z(t+1)/K4 ~ A * lam^2 z(t-1)/K4 + B * h4(t)/K4   (lam^2 z(t-1) = the old clock had nothing happened; exact gauge value of B is 1)")
for a in range(3, 9):
    q = f4 & (age == a)
    if q.sum() < 500: continue
    c, e = cfit(z[bb, tt + 1][q], [LAMC ** 2 * z[bb, tt - 1][q], H[bb, tt, 4][q] / K4])
    c1, e1 = cfit(z[bb, tt + 1][q], [LAMC ** 2 * z[bb, tt - 1][q]])
    print(f"   a={a}: A = {fmt(c[0])}  B = {fmt(c[1])}  EV {e:.4f}   (old clock alone: A = {fmt(c1[0])}, EV {e1:.4f})   rms kick/rms old clock = {np.sqrt(np.mean((H[bb, tt, 4][q] / K4) ** 2) / np.mean(abs(z[bb, tt - 1][q]) ** 2)):.3f}   n={q.sum()}")
print("for comparison, the same regression after a TRUE record (x_t > M): the old clock's coefficient")
bb2, tt2 = np.nonzero((x[:, 1:9] > M[:, :8]) & (M[:, :8] > 0)); tt2 = tt2 + 1
c, e = cfit(z[bb2, tt2 + 1], [LAMC ** 2 * z[bb2, tt2 - 1], H[bb2, tt2, 4] / K4])
print(f"   A = {fmt(c[0])}  B = {fmt(c[1])}  EV {e:.4f}   n={bb2.size}")
