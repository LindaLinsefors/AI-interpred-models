"""D1 (step 4). Geometry of the fitted vectors u(a), v(a), a = 2..9: PCA, plane vs clock plane, centre vs level eigenvector,
rotation per step, spiral fit; and R^2 of truncated descriptions on the data.   python -B d1_geometry.py [strict|pos] [regs]
"""
from d1_lib import *
from scipy.optimize import least_squares

rule = sys.argv[1] if len(sys.argv) > 1 else "strict"
regs = (sys.argv[2] if len(sys.argv) > 2 else "g,S,T,x").split(",")
G = load_gram(rule).sum(0)                 # (a1, a2+1, K, K)
Ga = G.sum(1)                              # (a1, K, K)
AGES = np.arange(2, 10)
Bfull = {a: fit(Ga[a], regs, YH)["B"] for a in range(10)}
WIDX = [ORD.index(u) for u in WAVE]
kg, kS = regs.index("g"), regs.index("S")
print(f"rule = {rule}; regressors = {regs}; ages {AGES[0]}..{AGES[-1]}")


def spiral_fit(Uv, ages):
    """Uv (n, 13) ~ c + Re(q lam^a).  Returns c, q, |lam|, angle (deg), residual."""
    def design(p):
        la = (p[0] * np.exp(1j * p[1])) ** ages
        return np.column_stack([np.ones(len(ages)), la.real, -la.imag])

    def resid(p):
        D = design(p)
        coef, *_ = np.linalg.lstsq(D, Uv, rcond=None)
        return (Uv - D @ coef).ravel()

    best = None
    for ang in np.radians([30, 42.75, 55, 83, 125]):
        r = least_squares(resid, [0.95, ang], bounds=([0.3, 0.05], [1.3, 3.1]))
        if best is None or r.cost < best.cost:
            best = r
    D = design(best.x)
    coef, *_ = np.linalg.lstsq(D, Uv, rcond=None)
    return coef[0], coef[1] + 1j * coef[2], best.x[0], np.degrees(best.x[1]), Uv - D @ coef


def analyse(name, k):
    Uv = np.stack([Bfull[a][k] for a in AGES])            # (8, 13)
    print(f"\n================ {name}(a), a = {AGES[0]}..{AGES[-1]} ================")
    mu = Uv.mean(0)
    _, s, Vt = np.linalg.svd(Uv - mu, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    tot = ((Uv - mu) ** 2).sum()
    print(f"PCA about the mean over ages: variance shares = {share[:5].round(4)}  (first two together {share[:2].sum():.4f})")
    print(f"   |mean| = {np.linalg.norm(mu):.2f};  rms distance from the mean = {np.sqrt(tot / len(AGES)):.2f};"
          f"  rms off the PC0-PC1 plane = {np.sqrt((s[2:] ** 2).sum() / len(AGES)):.2f}")
    V2 = Vt[:2].T
    print(f"   principal angles, PCA plane vs clock plane (deg): {principal_angles(V2, CLOCK_PLANE).round(1)}")
    print(f"   cos(mean, level eigvec) = {cosine(mu, LEVEL):.4f};  mean in mode coords/10.157: level {LEFT['level'].real @ mu / 10.157:.3f},"
          f" runner {LEFT['runner'].real @ mu / 10.157:.3f}")
    nperp = mu - V2 @ (V2.T @ mu)
    print(f"   perpendicular from the origin to the PCA plane: |n| = {np.linalg.norm(nperp):.2f}, cos(n, level eigvec) = {cosine(nperp, LEVEL):.4f}")
    P = (Uv - mu) @ V2
    ang = np.degrees(np.unwrap(np.arctan2(P[:, 1], P[:, 0])))
    print(f"   in-plane radius about the mean, per age: {np.linalg.norm(P, axis=1).round(2)}")
    print(f"   in-plane rotation per step about the mean (deg): {np.abs(np.diff(ang)).round(1)}   mean {np.abs(np.diff(ang)).mean():.2f}")
    c, q, lmod, lang, res = spiral_fit(Uv, AGES)
    print(f"spiral fit {name}(a) = c + Re(q lam^a): |lam| = {lmod:.4f}, angle = {lang:.2f} deg; residual rms = {np.sqrt((res ** 2).sum(1).mean()):.3f}"
          f"  (unexplained share of the variance about the mean = {(res ** 2).sum() / tot:.4f})")
    print(f"   centre c: |c| = {np.linalg.norm(c):.2f}, cos(c, level eigvec) = {cosine(c, LEVEL):.4f}; c in mode coords/10.157: "
          + ", ".join(f"{n} {abs(LEFT[n] @ c) / 10.157:.3f}" for n in NAMES))
    Qp = np.stack([q.real, q.imag], 1)
    rad = np.sqrt((abs(q) ** 2).sum() / 2)
    print(f"   spiral plane vs clock plane, principal angles (deg): {principal_angles(Qp, CLOCK_PLANE).round(1)};"
          f"  13-dim radius (phase-averaged) at a=2: {rad * lmod ** 2:.2f}, at a=9: {rad * lmod ** 9:.2f}")
    print(f"   centre c per unit {ORD}:\n     {c.round(2)}\n   radius |q_i| |lam|^2 per unit (a = 2):\n     {(abs(q) * lmod ** 2).round(2)}")
    print("   mode content: share of the variance about the mean (covariance share | own variance) and share of |mean|^2")
    dev = Uv - mu
    for n in NAMES:
        part = mode_parts(Uv, [n])
        pd = part - part.mean(0)
        print(f"     {n:7s} {(pd * dev).sum() / tot:8.4f} | {(pd ** 2).sum() / tot:7.4f}     mean: {part.mean(0) @ mu / (mu @ mu):8.4f}")
    return Uv, (c, q, lmod, lang)


Uu, sp_u = analyse("u", kg)
Uvv, sp_v = analyse("v", kS)

print("\n================ R^2 of truncated descriptions (states with a1 = 2..9, pooled) ================")
print("  description" + " " * 70 + "all 13 units: centred / uncentred | wave units: centred")


def evaluate(label, make_B):
    parts = [fit_given(Ga[a], regs, make_B(a), YH) for a in AGES]
    R = r2(parts)
    print(f"  {label:80s} {R['c_all']:.4f} / {R['u_all']:.4f} | {1 - R['var_res'][WIDX].sum() / R['var_y'][WIDX].sum():.4f}")


def trunc(a, modes_u, modes_v):
    B = Bfull[a].copy()
    B[kg], B[kS] = mode_parts(B[kg], modes_u), mode_parts(B[kS], modes_v)
    return B


def spiral_B(a, v_const=False):
    B = Bfull[a].copy()
    c, q, lm, la = sp_u
    B[kg] = c + np.real(q * (lm * np.exp(1j * np.radians(la))) ** a)
    c, q, lm, la = sp_v
    B[kS] = c if v_const else c + np.real(q * (lm * np.exp(1j * np.radians(la))) ** a)
    return B


evaluate("free u(a), v(a) (least squares)", lambda a: Bfull[a])
evaluate("u, v -> level + runner + clock + echo parts only", lambda a: trunc(a, ["level", "runner", "clock", "echo"], ["level", "runner", "clock", "echo"]))
evaluate("u, v -> level + clock + echo", lambda a: trunc(a, ["level", "clock", "echo"], ["level", "clock", "echo"]))
evaluate("u -> level + runner + clock + echo; v -> level + runner + echo (no S rotation)", lambda a: trunc(a, ["level", "runner", "clock", "echo"], ["level", "runner", "echo"]))
evaluate("u, v -> level + runner + echo (no rotation at all)", lambda a: trunc(a, ["level", "runner", "echo"], ["level", "runner", "echo"]))
evaluate("u = c_u + spiral, v = c_v + spiral (the two spiral fits above)", lambda a: spiral_B(a))
evaluate("u = c_u + spiral, v = c_v (constant)", lambda a: spiral_B(a, True))
