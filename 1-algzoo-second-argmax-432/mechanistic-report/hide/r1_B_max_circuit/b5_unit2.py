"""B5. Unit 2: is it the max integrator? Is the level mode mostly unit 2?
Data: N(0,1), seeds 1050, 1051 (400k each).
"""
from blib import *

i2 = OI[2]
print("=== [EXACT] weights into unit 2 ===")
print("  w_2 =", round(w[2], 4))
print("  W[2, :] =", " ".join(f"{u}:{W[2, u]:+.3f}" for u in range(N)))
print("  record path: W[2,4]*w_4 =", round(W[2, 4] * W4, 3), "per unit of (x - theta_4);  threshold path: -W[4,2]/w_4 =", round(-W[4, 2] / W4, 5),
      "; product W[2,4]*(-W[4,2]) =", round(-W[2, 4] * W[4, 2], 4), "(threshold gained per unit record via unit 2 alone)")
print("\n=== [EXACT] unit 2 in the modes: LEFT_n[2] (what a unit-2 perturbation excites), RIGHT_n[2] (how much of each mode shows in unit 2) ===")
for n in NAMES:
    l, r = LEFT[n][i2], RIGHT[n][i2]
    if IS_COMPLEX[n]:
        print(f"  {n:7s}: |LEFT[2]| = {abs(l):.4f} (share of |LEFT|^2: {abs(l)**2/np.sum(abs(LEFT[n])**2):.3f});  2|RIGHT[2]| = {2*abs(r):.4f};  2 Re(L*R) = {2*(l*r).real:+.4f}")
    else:
        print(f"  {n:7s}: LEFT[2] = {l.real:+.4f} (share of |LEFT|^2: {l.real**2/np.sum(LEFT[n].real**2):.3f});  RIGHT[2] = {r.real:+.4f};  L*R = {(l*r).real:+.4f}")
print("  sum over modes of L*R (must be 1):", sum((2 * (LEFT[n][i2] * RIGHT[n][i2]).real if IS_COMPLEX[n] else (LEFT[n][i2] * RIGHT[n][i2]).real) for n in NAMES))
print("  cosine between LEFT_level and e_2:", LEFT["level"].real[i2] / np.linalg.norm(LEFT["level"].real),
      "; cosine between RIGHT_level and e_2:", RIGHT["level"].real[i2] / np.linalg.norm(RIGHT["level"].real))
print("  eigen-decomposition of the 1x1 'integrator' W[2,2] =", round(W[2, 2], 4), "vs lam_level =", round(LAM["level"].real, 4))

SEEDS = (1050, 1051)
P = []
for seed in SEEDS:
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = prefix_stats(x)
    P.append(dict(x=x, H=H, z2=Z[:, :, 2], M=st["M1"], S=st["M2"], a=st["a1"]))
x, H, z2, M, S, a = (np.concatenate([p[k] for p in P]) for k in ("x", "H", "z2", "M", "S", "a"))
Sp = pos(np.where(np.isfinite(S), S, 0))
n = len(x)
tt = np.broadcast_to(np.arange(10), x.shape)
h2 = H[:, :, 2]
m = coords(H)
lev = m["level"]
print(f"\n=== [MEASURED n = {n} sequences, N(0,1), seeds {SEEDS}] ===")

print("\n--- h_2 against M by age of the max (states with M > 0): h_2/10.157 ---")
print("age | n | median h2/(10.157 M) | IQR | slope (no icpt) | R2c | slopes on (M, S+) | R2c | P(unit 2 off) | level slope for comparison")
for age in range(10):
    s = (a == age) & (M > 0)
    q = h2[s] / (W4 * M[s])
    b1, _, r1 = slope0(h2[s] / W4, [M[s]])
    b2, _, r2 = slope0(h2[s] / W4, [M[s], Sp[s]])
    bl, _, _ = slope0(lev[s] / W4, [M[s]])
    print(f"{age} | {s.sum()} | {np.median(q):.4f} | [{np.percentile(q,25):.4f}, {np.percentile(q,75):.4f}] | {b1[0]:.4f} | {r1:.4f} | {b2[0]:.4f}, {b2[1]:+.4f} | {r2:.4f} | {(z2[s] <= 0).mean():.5f} | {bl[0]:.4f}")
s = (M > 0) & (a >= 1)
b, _, r = slope0(h2[s] / W4, [M[s]])
print(f"  all ages >= 1 pooled: h2/10.157 = {b[0]:.4f} M, R2c = {r:.4f};  affine fit R2: {np.corrcoef(h2[s], M[s])[0,1]**2:.4f}")
print(f"  P(unit 2 off) overall: {(z2 <= 0).mean():.4f};  given M <= 0: {(z2[M <= 0] <= 0).mean():.4f};  given M > 0: {(z2[M > 0] <= 0).mean():.5f}")

print("\n--- what feeds unit 2: slope (through 0, on M) of each term of z_2(t) = w_2 x_t + sum_u W[2,u] h_u(t-1), divided by 10.157; by age of the max at t (M > 0, t >= 1) ---")
Hprev = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
srcs = {"self 0.969*h2": [2], "unit 4": [4], "unit 9": [9], "unit 5": [5], "comparators 1,6,7,8": COMP, "other wave units": [u for u in WAVE if u != 2]}
print("age | h2(t) | " + " | ".join(srcs) + " | input w_2 x_t | sum | h2(t) - h2(t-1) | of which leak -0.031*h2(t-1) | comparators | other waves | unit 9")
for age in range(10):
    s = (a == age) & (M > 0) & (tt >= 1)
    den = (M[s] ** 2).sum() * W4
    sl = lambda y: (y[s] * M[s]).sum() / den
    vals = [sl(Hprev[:, :, us] @ W[2, us]) for us in srcs.values()]
    vx = sl(w[2] * x)
    print(f"{age} | {sl(h2):.4f} | " + " | ".join(f"{v:+.4f}" for v in vals) + f" | {vx:+.4f} | {sum(vals)+vx:.4f} | {sl(h2 - Hprev[:, :, 2]):+.4f} | "
          f"{sl((W[2,2]-1)*Hprev[:, :, 2]):+.4f} | {vals[4]:+.4f} | {vals[5]:+.4f} | {vals[2]:+.4f}")

print("\n--- per source unit: slope of W[2,u]*h_u(t-1)/10.157 on M, ages 3..9 pooled (resting phase) ---")
s = (a >= 3) & (M > 0)
den = (M[s] ** 2).sum() * W4
for u in range(N):
    print(f"  from unit {u:2d}: W[2,u] = {W[2,u]:+.3f}, term slope {((Hprev[:, :, u][s] * W[2, u]) * M[s]).sum()/den:+.4f}")

print("\n--- leaky-integrator test: g(t) = W[2,2]*g(t-1) + W[2,4]*h4(t-1) (unit 2 fed by unit 4 only), compared with the real h_2 ---")
g = np.zeros_like(h2)
for t in range(1, 10):
    g[:, t] = W[2, 2] * g[:, t - 1] + W[2, 4] * H[:, t - 1, 4]
g1 = np.zeros_like(h2)                       # the same with no leak
for t in range(1, 10):
    g1[:, t] = g1[:, t - 1] + W[2, 4] * H[:, t - 1, 4]
print("t | corr^2(h2, leaky) | slope h2 ~ leaky | corr^2(h2, no-leak) | slope h2 ~ no-leak")
for t in range(1, 10):
    c1 = np.corrcoef(h2[:, t], g[:, t])[0, 1] ** 2
    c2 = np.corrcoef(h2[:, t], g1[:, t])[0, 1] ** 2
    print(f"{t} | {c1:.4f} | {(h2[:, t]*g[:, t]).sum()/(g[:, t]**2).sum():.4f} | {c2:.4f} | {(h2[:, t]*g1[:, t]).sum()/(g1[:, t]**2).sum():.4f}")

print("\n--- unit 2's share of the level coordinate, L_2*h_2 / level, and the level mode's share of h_2, R_level[2]*level / h_2 (M > 0) ---")
L2, R2 = LEFT["level"].real[i2], RIGHT["level"].real[i2]
print("age | sum(L_2 h_2 M)/sum(level M) | median L_2 h_2/level | sum(R_2 level M)/sum(h_2 M) | h_2 split over modes (slope on M /10.157): " + ", ".join(NAMES))
for age in range(10):
    s = (a == age) & (M > 0)
    den = (M[s] ** 2).sum() * W4
    split = []
    for nme in NAMES:
        c = 2 * np.real(m[nme][s] * RIGHT[nme][i2]) if IS_COMPLEX[nme] else m[nme][s] * RIGHT[nme][i2].real
        split.append((c * M[s]).sum() / den)
    print(f"{age} | {(L2*h2[s]*M[s]).sum()/(lev[s]*M[s]).sum():.4f} | {np.median(L2*h2[s]/lev[s]):.4f} | {(R2*lev[s]*M[s]).sum()/(h2[s]*M[s]).sum():.4f} | " + ", ".join(f"{v:+.3f}" for v in split))

print("\n--- level coordinate split by unit: slope of L_u*h_u/10.157 on M (ages >= 2, M > 0) ---")
s = (a >= 2) & (M > 0)
den = (M[s] ** 2).sum() * W4
tot = (lev[s] * M[s]).sum() / den
for i, u in enumerate(ORD):
    v = (LEFT["level"].real[i] * H[:, :, u][s] * M[s]).sum() / den
    print(f"  unit {u:2d}: {v:+.4f}  ({v/tot:+.3f} of the level coordinate {tot:.4f})")

print("\n--- theta_4 split by source unit: slope of -W[4,u] h_u(t-1)/w_4 on M_{t-1}, by age of the max at t-1 (M > 0) ---")
Mx, ax = M[:, :-1], a[:, :-1]
Hp = H[:, :-1]
groups = {"unit 2": [2], "unit 4": [4], "unit 9": [9], "comparators": COMP, "other wave": [u for u in WAVE if u != 2], "unit 5": [5]}
print("age | theta_4 | " + " | ".join(groups))
for age in range(9):
    s = (ax == age) & (Mx > 0)
    den = (Mx[s] ** 2).sum()
    th = -(Hp @ W[4]) / W4
    print(f"{age} | {(th[s]*Mx[s]).sum()/den:.4f} | " + " | ".join(f"{((-(Hp[:, :, us] @ W[4, us]) / W4)[s]*Mx[s]).sum()/den:+.4f}" for us in groups.values()))

print("\n--- [INTERVENTION] scale unit 2 by f at step t0 (h_2 := f*h_2) and read theta_4 at t0+1, t0+2, t0+4: change relative to M (median), states with M > 0, age >= 2; n = 200k, seed 1052 ---")
xi = sample(200_000, 1052)
Hi = run(xi)
sti = prefix_stats(xi)
print("t0 | f | exact d theta_4(t0+1)/M = 0.0484*(f-1)*h2/M (median) | measured (median) | d theta_4(t0+2)/M | d theta_4(t0+4)/M | level-mode prediction 0.7956*L_2*(f-1)*h2*lam^k/(10.157 M) at k = 1, 3")
for t0 in (3, 5):
    for f in (0.8, 1.2):
        h0 = Hi[:, t0].copy()
        h0[:, 2] *= f
        Hp_ = run(xi[:, t0 + 1:], h0=h0)
        sel = (sti["M1"][:, t0] > 0) & (sti["a1"][:, t0] >= 2)
        Mi = sti["M1"][:, t0]
        d = lambda k: np.median(((theta4(Hp_[:, k - 1]) - theta4(Hi[:, t0 + k]))[sel]) / Mi[sel])   # theta for input t0+k+1 from state t0+k
        d1 = np.median((0.492 / W4 * (f - 1) * Hi[:, t0, 2] / Mi)[sel])
        direct = np.median(((theta4(h0) - theta4(Hi[:, t0]))[sel]) / Mi[sel])
        pred = lambda k: np.median((C4["level"].real * LEFT["level"].real[i2] * (f - 1) * Hi[:, t0, 2] * LAM["level"].real ** k / Mi)[sel])
        print(f"{t0} | {f} | {d1:+.4f} | {direct:+.4f} | {d(1):+.4f} | {d(3):+.4f} | {pred(1):+.4f}, {pred(3):+.4f}")
