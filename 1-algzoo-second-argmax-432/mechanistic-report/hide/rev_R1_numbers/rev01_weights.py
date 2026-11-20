"""Reviewer R1 (numbers audit): everything in report.md that follows from the weights alone.
Run:  python -B rev01_weights.py > rev01_weights_out.txt
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import LEFT, RIGHT, LAM, NAMES, IS_COMPLEX, coords

np.set_printoptions(precision=4, suppress=True, linewidth=220)
A = W[np.ix_(ORD, ORD)]
Uo = U[:, ORD]
K = w[4]
deg = lambda z: np.degrees(np.angle(z))

print("== section 2: input weights")
print("  comparators", np.round(w[COMP], 3), " w4", round(w[4], 4), " w5, w9", np.round(w[[5, 9]], 3))
print("  wave units", dict(zip(WAVE, np.round(w[WAVE], 3))), " max |w| without unit 14:", round(max(abs(w[u]) for u in WAVE if u != 14), 3))

print("\n== section 3.2: Sigma")
Sig = Uo @ A @ np.linalg.pinv(Uo)
print("  relative residual", round(np.linalg.norm(Uo @ A - Sig @ Uo) / np.linalg.norm(Uo @ A), 4))
sup = [Sig[i, i + 1] for i in range(9)]
print("  superdiagonal", np.round(sup, 3))
off = Sig.copy()
for i in range(9):
    off[i, i + 1] = 0
print("  largest |other entry| in rows 0-6:", round(np.abs(off[:7]).max(), 3), "at", np.unravel_index(np.abs(off[:7]).argmax(), off[:7].shape))
print("  largest |other entry| in rows 0-8:", round(np.abs(off[:9]).max(), 3), "at", np.unravel_index(np.abs(off[:9]).argmax(), off[:9].shape))
print("  row 7:", np.round(Sig[7], 2))
print("  row 8:", np.round(Sig[8], 2))
print("  row 9:", np.round(Sig[9], 2), " mean", round(Sig[9].mean(), 3), " min/max", round(Sig[9].min(), 2), round(Sig[9].max(), 2))
lam = np.linalg.eigvals(A)
print("  eigenvalues of A: |lam|, angle:", sorted([(round(abs(l), 4), round(float(deg(l)), 2)) for l in lam], reverse=True))
print("  number of negative entries of A:", int((A < 0).sum()), "of", A.size)

print("\n== section 3.3 / 6: slopes")
print("  U[9,COMP]*w[COMP] =", np.round(U[9, COMP] * w[COMP], 3), " sum", round((U[9, COMP] * w[COMP]).sum(), 3))
print("  U[9,4]*w4 =", round(U[9, 4] * w[4], 3), "  U[9,4] =", round(U[9, 4], 3), "  U[8,5] =", round(U[8, 5], 4), "  U[8,9] =", round(U[8, 9], 3))
c5 = W[5, ORD] @ w[ORD]
print("  W[5,ORD].w[ORD] =", round(c5, 4), " -> times U[8,5] =", round(c5 * U[8, 5], 3))
print("  W[5,[7,8]].w[[7,8]] (two comparators only) =", round(W[5, [7, 8]] @ w[[7, 8]], 4), " -> times U[8,5] =", round((W[5, [7, 8]] @ w[[7, 8]]) * U[8, 5], 3))
print("  W[5,COMP].w[COMP] (four comparators) =", round(W[5, COMP] @ w[COMP], 4))
Kx1 = Uo @ A @ w[ORD]
print("  raw-input kernel one step later (U_o A w_o), positions 0..9:", np.round(Kx1, 3))
print("  total d logit[8] / d x_8 when unit 5 is on and the 13 ordinary units are on: U[8,5]*18.009 + (U_o A w_o)[8] =", round(c5 * U[8, 5] + Kx1[8], 3))
dl = c5 * U[:, 5] + Kx1
print("  d(L_8 - L_p)/dx_8 for p != 8 (same pattern):", np.round(dl[8] - np.delete(dl, 8), 2))
print("  L_9 coefficients: U[9,j]*g_j =", np.round(U[9, COMP] * (-w[COMP]), 3), " U[9,4]*w4 =", round(U[9, 4] * w[4], 3))
# slopes of L_9 - L_p in x_9, all 13 ordinary on / comparators clipped + unit 4 on
below = (U[9] - U) @ np.where(np.isin(np.arange(16), ORD), w, 0)
above = (U[9] - U) @ np.where(np.isin(np.arange(16), WAVE + [4]), w, 0)
print("  d(L_9 - L_p)/dx_9, p = 0..8, below all thresholds:", np.round(below[:9], 2))
print("  d(L_9 - L_p)/dx_9, p = 0..8, above all thresholds:", np.round(above[:9], 2))

print("\n== section 4.1: core wiring")
g = -w[COMP]
Bc = W[np.ix_(COMP, COMP)] * g[None, :] / g[:, None]
print("  Bc =\n", np.round(Bc, 3))
print("  Bc 1 =", np.round(Bc.sum(1), 4))
ev = np.linalg.eigvals(Bc)
print("  eig Bc:", [(round(abs(e), 4), round(float(deg(e)), 2)) for e in ev])
print("  b2 =", np.round(W[COMP, 2] / g, 4), "  W[j,2]/w_j =", np.round(W[COMP, 2] / w[COMP], 4))
print("  w4*b4 =", np.round(w[4] * W[COMP, 4] / g, 4))
print("  unit 2: W[2,2] =", round(W[2, 2], 4), " W[2,4] =", round(W[2, 4], 4), " w2 =", round(w[2], 4))
print("  theta4' coefficients: -W[4,2]/w4 =", round(-W[4, 2] / w[4], 4), "  -W[4,4]/w4 =", round(-W[4, 4] / w[4], 4), "  -W[4,4] =", round(-W[4, 4], 4))
cs = [W[u, COMP] @ w[COMP] / (np.linalg.norm(W[u, COMP]) * np.linalg.norm(w[COMP])) for u in range(16)]
print("  cos(row on comparators, comparators' input weights):", dict(zip(range(16), np.round(cs, 3))))
print("  max |cos| over the 14 other units:", round(max(abs(c) for u, c in enumerate(cs) if u not in (5, 9)), 4))
left = W[:, ORD] @ w[ORD]
print("  W[:,ORD] w[ORD] (13 ordinary on):", dict(zip(range(16), np.round(left, 3))))
left4 = W[:, COMP] @ w[COMP]
print("  W[:,COMP] w[COMP] (only the four comparators' share):", dict(zip(range(16), np.round(left4, 3))))

print("\n== section 4.2: weights quoted")
print("  U[7,0], U[5,12], U[4,11], U[3,10] =", np.round([U[7, 0], U[5, 12], U[4, 11], U[3, 10]], 3))
for u in (0, 12, 11, 10):
    print(f"    unit {u}: most negative readout weight {U[:, u].min():.3f} at position {U[:, u].argmin()}")
print("  W[0,5], W[15,5], W[3,5], W[14,5], W[12,5] =", np.round([W[0, 5], W[15, 5], W[3, 5], W[14, 5], W[12, 5]], 3), "  W[0,9] =", round(W[0, 9], 3))
print("  W[WAVE,5] =", dict(zip(WAVE, np.round(W[WAVE, 5], 3))))

print("\n== section 4.3: modes")
for n in NAMES:
    l = LAM[n]
    L2 = np.abs(LEFT[n]) ** 2
    share_comp = L2[[ORD.index(u) for u in COMP]].sum() / L2.sum()
    share_u2 = L2[ORD.index(2)] / L2.sum()
    R = Uo @ RIGHT[n]
    def seen(v):
        v = v[:9] - v[:9].mean()
        return np.linalg.norm(v)
    s = (seen(R.real), seen(R.imag)) if IS_COMPLEX[n] else (seen(R.real),)
    # the report's 'seen by the readout' uses the real basis vectors 2Re, 2Im? print both conventions
    s2 = tuple(2 * x for x in s) if IS_COMPLEX[n] else s
    print(f"  {n:7s} |lam| {abs(l):.4f} angle {float(deg(l)):7.2f}  left: comparators {share_comp:.3f}, unit 2 {share_u2:.3f}, wave units {1 - share_comp:.3f}   seen (pos 0-8, mean removed): per Re/Im {np.round(s, 2)}  x2: {np.round(s2, 2)}")
lv = RIGHT["level"].real
print("  level right vector / unit 2:", dict(zip(ORD, np.round(lv / lv[ORD.index(2)], 3))))
echo = RIGHT["echo"].real
print("  cos(w[ORD], echo eigenvector) =", round(abs(w[ORD] @ echo) / np.linalg.norm(w[ORD]) / np.linalg.norm(echo), 4))
ic = [ORD.index(u) for u in COMP]; iw = [ORD.index(u) for u in WAVE]
Rc = U[:, COMP] @ RIGHT["clock"][ic]
Rw = U[:, WAVE] @ RIGHT["clock"][iw]
print("  clock readout halves, positions 0..9:")
print("    comparators 2|R|:", np.round(2 * np.abs(Rc), 2), " phase", np.round(deg(Rc), 1))
print("    wave units  2|R|:", np.round(2 * np.abs(Rw), 2), " phase", np.round(deg(Rw), 1))
dph = np.abs((deg(Rc) - deg(Rw) + 180) % 360 - 180)
print("    phase difference (deg):", np.round(dph, 1), "  range over positions 0-7:", round(dph[:8].min(), 1), "-", round(dph[:8].max(), 1))
tot = Rc + Rw
print("    sum 2|R|:", np.round(2 * np.abs(tot), 2))
print("    ratio |sum| / (|comp| + |wave|): all 10 positions", round(np.linalg.norm(tot) / (np.linalg.norm(Rc) + np.linalg.norm(Rw)), 4),
      "  positions 0-7:", round(np.linalg.norm(tot[:8]) / (np.linalg.norm(Rc[:8]) + np.linalg.norm(Rw[:8])), 4),
      "  per position:", np.round(np.abs(tot) / (np.abs(Rc) + np.abs(Rw)), 3))
Rl = Uo @ RIGHT["level"].real
print("  level readout per unit coordinate, positions 0..9:", np.round(Rl, 3))

print("\n== section 5.1 / 5.2 / 5.4: exact coefficients")
print("  LEFT_level . w[ORD] =", round((LEFT["level"] @ w[ORD]).real, 4), "   LEFT_runner . w[ORD] =", round((LEFT["runner"] @ w[ORD]).real, 4),
      "   (kick of a new max of excess d writes 10.157 d; ratio runner:", round((LEFT["runner"] @ w[ORD]).real / K, 4), " level:", round((LEFT["level"] @ w[ORD]).real / K, 5), ")")
print("  level update coefficients: LEFT_level.W[ORD,5] =", round((LEFT["level"] @ W[ORD, 5]).real, 4), "  LEFT_level.W[ORD,9] =", round((LEFT["level"] @ W[ORD, 9]).real, 4))
# Theta_5 in mode coordinates: 18.009 Theta5 = -W[5,ORD] A h[ORD] ... coefficient per unit mode coordinate
for n in ("level", "runner", "clock"):
    c = -(W[5, ORD] @ A @ RIGHT[n]) / c5
    print(f"  Theta_5 coefficient per unit {n} coordinate: {np.round(c if not IS_COMPLEX[n] else 2*abs(c), 5)}")
print("  W[5,4]*w4 =", round(W[5, 4] * K, 3), "  W[9,4]*w4 =", round(W[9, 4] * K, 3), "  W[5,4] =", round(W[5, 4], 4), " W[9,4] =", round(W[9, 4], 4))
print("  W[5,7]*w7, W[5,8]*w8 =", round(W[5, 7] * w[7], 3), round(W[5, 8] * w[8], 3))
c5c = LEFT["clock"] @ W[ORD, 5]; c9c = LEFT["clock"] @ W[ORD, 9]
t1 = LAM["clock"]; t2 = c5c * W[5, 4]; t3 = c9c * W[9, 4]
print(f"  first step of the clock: lam = {abs(t1):.3f}@{float(deg(t1)):.1f}  c5 W[5,4] = {abs(t2):.3f}@{float(deg(t2)):.1f}  c9 W[9,4] = {abs(t3):.3f}@{float(deg(t3)):.1f}  sum = {abs(t1+t2+t3):.3f}@{float(deg(t1+t2+t3)):.1f}")
print("  unit-5 write vs unit 4's one-step-rotated kick, per mode: |c5|, arg c5 - arg lam:")
for n in NAMES:
    c = LEFT[n] @ W[ORD, 5]
    print(f"    {n:7s} c5 = {abs(c):.3f}@{float(deg(c)):7.1f}   arg(c5) - arg(lam) = {float((deg(c) - deg(LAM[n]) + 180) % 360 - 180):7.1f}   c9 = {abs(LEFT[n] @ W[ORD, 9]):.3f}@{float(deg(LEFT[n] @ W[ORD, 9])):7.1f}")

print("\n== impulse: level bookkeeping from rest (units of 10.157 M)")
x = np.zeros((1, 10)); x[0, 0] = 1
H, Z = run(x, return_pre=True)
m = coords(H)
print("  level/K by t:", np.round(m["level"][0] / K, 4))
print("  |clock|/K by t:", np.round(np.abs(m["clock"][0]) / K, 4), " phase:", np.round(deg(m["clock"][0]), 1))
print("  runner/K by t:", np.round(m["runner"][0] / K, 3))
print("  h5(1)/K, h9(1)/K:", round(H[0, 1, 5] / K, 4), round(H[0, 1, 9] / K, 4), "  -> 0.0724*h5 =", round(0.0724 * H[0, 1, 5] / K, 4), " 0.1263*h9 =", round(0.1263 * H[0, 1, 9] / K, 4))
print("  1.006*1.0099 - 0.087 - 0.019 =", round(1.0063 * 1.00989 - 0.0870 - 0.0186, 4), " (the report's two subtractions give this, not 0.927)")
print("  h2/M by t:", np.round(H[0, :, 2], 2), "  theta_4 for the next input by t:", np.round(-(H[0] @ W[4]) / w[4], 3))
print("  growth of level per step, t=3..9:", np.round(m["level"][0, 3:] / m["level"][0, 2:-1], 4))

print("\n== two exact two-pulse examples (section 8): |z|/K against rho(a)(M-S), rho(a) = 1.02*0.951^(a-2)")
for xs in ([1, 0, 0, 0, 2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 1, 0, 0, 0, 0, 0]):
    H = run(np.array([xs], float))
    z = coords(H)["clock"][0] / K
    pM = int(np.argmax(xs))
    out = []
    for t in range(10):
        a = t - pM
        if a >= 2 and t >= 6:
            out.append((t, a, round(abs(z[t]), 3), round(abs(z[t]) / (1.02 * 0.951 ** (a - 2) * 1.0), 3), round(float(deg(z[t])), 1)))
    print("  x =", xs, " (t, age of max, |z|/K, ratio to rho(a)(M-S), phase):", out)

print("\n== runner right eigenvector in the gauge (state per unit runner coordinate) times 10.157*5.2 = change of centre per unit S")
rr = RIGHT["runner"].real
print("  ", dict(zip(ORD, np.round(rr * K * 5.2, 2))))
print("   same with r1 = 5.0:", dict(zip(ORD, np.round(rr * K * 5.0, 2))))

print("\n== section 3.4: -60 cos(42.75 d) and c(D)/(1-cos(42.75 D))")
print("  -60cos:", np.round(-60 * np.cos(np.radians(42.75 * np.arange(8))), 1))
c = np.array([17.6, 43.5, 73.8, 88.5, 81.6, 63.2, 41.8])
print("  1-cos(42.75 D), D=1..8:", np.round(1 - np.cos(np.radians(42.75 * np.arange(1, 9))), 3))
print("  c/(1-cos):", np.round(c / (1 - np.cos(np.radians(42.75 * np.arange(1, 8)))), 1))
print("  period 360/42.75 =", round(360 / 42.75, 3))
print("  sinusoid fit of the comparators' readout columns, p = 0..7 (frequency fixed at 42.748 deg):")
om = np.angle(LAM["clock"])
p = np.arange(8)
for j in COMP:
    X = np.column_stack([np.cos(om * p), np.sin(om * p), np.ones(8)])
    co, *_ = np.linalg.lstsq(X, U[:8, j], rcond=None)
    r2 = 1 - ((U[:8, j] - X @ co) ** 2).sum() / ((U[:8, j] - U[:8, j].mean()) ** 2).sum()
    X9 = np.column_stack([np.cos(om * np.arange(9)), np.sin(om * np.arange(9)), np.ones(9)])
    co9, *_ = np.linalg.lstsq(X9, U[:9, j], rcond=None)
    r29 = 1 - ((U[:9, j] - X9 @ co9) ** 2).sum() / ((U[:9, j] - U[:9, j].mean()) ** 2).sum()
    print(f"    unit {j}: a = {np.hypot(co[0], co[1]):.2f}, phi = {np.degrees(np.arctan2(co[1], co[0])):.0f} deg, c = {co[2]:.2f}, R2 (p=0..7) = {r2:.3f};  R2 if p = 0..8 is fitted: {r29:.3f};  U[8,j] = {U[8, j]:.2f} vs extrapolated {np.array([np.cos(om*8), np.sin(om*8), 1]) @ co:.2f}")
