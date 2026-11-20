"""V4 verifier: claims K0 (identity), K1 (core weights), K2 (cosines). Seed 5001."""
from vsetup import *

out = Tee("v1_k0_k1_k2.txt")
np.set_printoptions(precision=4, suppress=True, linewidth=200)

out("w[COMP] =", w[COMP], " (all negative:", bool((w[COMP] < 0).all()), ")")
out("w[4] =", w[4], " w[2] =", w[2])
out("w (all 16) =", w)

# ---------------------------------------------------------------- K0
out("\n=== K0: theta(t+1) = Bc [theta(t) - x_t]_+ + b2 h2(t) + b4 h4(t) + other ten units ===")
for dist in ("normal", "uniform"):
    x = sample(200_000, 5001, dist)
    H = run(x)
    th, th4 = thresholds(H)                                    # th[:, t] = theta(t), t = 0..10
    worst = 0.0
    worst_rel = 0.0
    worst_relu = 0.0
    for t in range(10):
        relu = np.maximum(th[:, t] - x[:, t, None], 0.0)       # [theta(t) - x_t 1]_+
        # the comparators' activations must equal g * relu
        worst_relu = max(worst_relu, np.abs(H[:, t][:, COMP] - g * relu).max())
        other = (H[:, t][:, OTHER10] @ W[np.ix_(COMP, OTHER10)].T) / g
        rhs = relu @ Bc.T + np.outer(H[:, t, 2], b2) + np.outer(H[:, t, 4], b4) + other
        err = np.abs(th[:, t + 1] - rhs)
        worst = max(worst, err.max())
        worst_rel = max(worst_rel, (err / (np.abs(th[:, t + 1]) + 1e-12)).max())
    out(f"{dist:8s} n=200000 seed 5001: max |lhs - rhs| over t=0..9, j: {worst:.3e}; "
        f"max |h_j - g_j[theta_j - x]_+|: {worst_relu:.3e}; typical |theta| = {np.abs(th[:, 1:]).mean():.3f}; "
        f"max |theta| = {np.abs(th).max():.2f}")
out("OTHER10 =", OTHER10)
out("float64 eps * max|theta| * ~50 terms ~ 1e-13: the identity is exact up to rounding.")

# ---------------------------------------------------------------- K1
out("\n=== K1 ===")
out("Bc =\n", Bc)
out("Bc @ 1 =", Bc @ np.ones(4))
ev, evec = np.linalg.eig(Bc)
for e in ev:
    out(f"  eigenvalue {e:.5f}: |.| = {abs(e):.4f}, angle = {np.degrees(np.angle(e)):8.3f} deg")
out("singular values of Bc:", np.linalg.svd(Bc, compute_uv=False))
out("b2 =", b2)
out("w4 * b4 =", w[4] * b4)
out("W[2,2] =", W[2, 2], " W[2,4] =", W[2, 4], " w2 =", w[2])
out("-W[4,2]/w4 =", -W[4, 2] / w[4], " -W[4,4]/w4 =", -W[4, 4] / w[4])
out("W[2,4]*w4 =", W[2, 4] * w[4], " (h2 gain per unit increment);  b2 * W[2,4] * w4 =", b2 * W[2, 4] * w[4])
out("-W[4,2]/w4 * W[2,4]*w4 =", -W[4, 2] * W[2, 4])

claim_Bc = np.array([[0.409, 1.036, -1.306, -0.140], [-0.130, 0.361, -0.151, -0.092],
                     [-0.081, -0.152, 0.078, 0.156], [0.544, 0.000, -1.170, 0.615]])
out("max |Bc - claimed| =", np.abs(Bc - claim_Bc).max())
out("max |Bc1 - claimed| =", np.abs(Bc @ np.ones(4) - np.array([-0.001, -0.012, 0.001, -0.011])).max())
out("max |b2 - claimed| =", np.abs(b2 - np.array([0.048, 0.049, 0.051, 0.051])).max())
out("max |w4 b4 - claimed| =", np.abs(w[4] * b4 - np.array([0.838, -0.044, 1.016, -0.034])).max())

# ---------------------------------------------------------------- K2
out("\n=== K2: cos( W[u, COMP], w[COMP] ) ===")
cosv = np.array([W[u, COMP] @ w[COMP] / (np.linalg.norm(W[u, COMP]) * np.linalg.norm(w[COMP])) for u in range(16)])
for u in range(16):
    grp = "COMP" if u in COMP else ("PULSE" if u in PULSE else "WAVE")
    out(f"  unit {u:2d} ({grp:5s}): cos = {cosv[u]:+.4f}   |W[u,COMP]| = {np.linalg.norm(W[u, COMP]):.4f}   "
        f"W[u,COMP].w[COMP] = {W[u, COMP] @ w[COMP]:+.4f}")
oth = [u for u in range(16) if u not in (5, 9)]
out("max |cos| over the 14 units other than 5, 9:", np.abs(cosv[oth]).max(), "at unit", oth[int(np.abs(cosv[oth]).argmax())])
out("cos unit 5 =", cosv[5], " cos unit 9 =", cosv[9])
