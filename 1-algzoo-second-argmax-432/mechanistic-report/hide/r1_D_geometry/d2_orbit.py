"""D2. Per-unit orbit of the max's age at fixed S/M: centre, clock radius, minimum over ages, measured clipping.

State used: the pre-input state s_t = W h_{t-1} (= pre-activation minus w x_t), in units of M.  For a comparator j,
s_j = |w_j| theta_j (its threshold times its input weight); for a wave unit s_i is the pre-activation up to |w_i x_t| <= 0.3|x_t|
(unit 14: 1.32 |x_t|).
  y(a) = least squares of s_t on M (no intercept) among states with a1 = a and S+/M in a narrow bin (S+ = max(S, 0)).
Decomposition in the modes of A:  centre = level + runner + trans + echo parts,  clock = 2 Re(z R_clock),  other = h2, alt, h3, short.
N(0,1), seeds 1041..1044 (4 x 200k).     python -B d2_orbit.py
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide")
from common import *
from modes import *
from d1_lib import load_gram, fit, YZ, mode_parts

np.set_printoptions(precision=2, suppress=True, linewidth=230)
oi = np.array(ORD)
BINS = {"0": (-1e-9, 0.05), "0.5": (0.45, 0.55), "0.9": (0.85, 0.95)}
AG = list(range(2, 10))
num = {b: np.zeros((10, 13)) for b in BINS}
den = {b: np.zeros(10) for b in BINS}
cnt = {b: np.zeros(10) for b in BINS}
off = {b: np.zeros((10, 13)) for b in BINS}          # actual gate off (pre-activation <= 0)
neg = {b: np.zeros((10, 13)) for b in BINS}          # pre-input state <= 0
sq = {b: np.zeros((10, 13)) for b in BINS}
for seed in range(1041, 1045):
    x = sample(200_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    M, S, a1 = st["M1"], st["M2"], st["a1"]
    r = np.maximum(np.where(np.isfinite(S), S, 0), 0) / np.where(M > 0, M, 1)
    Spre = (Z - x[..., None] * w)[..., oi]
    for b, (lo, hi) in BINS.items():
        for a in AG:
            s = (a1 == a) & (M > 0) & (r > lo) & (r <= hi)
            num[b][a] += (Spre[s] * M[s][:, None]).sum(0)
            den[b][a] += (M[s] ** 2).sum()
            cnt[b][a] += s.sum()
            off[b][a] += (Z[s][:, oi] <= 0).sum(0)
            neg[b][a] += (Spre[s] <= 0).sum(0)
            sq[b][a] += (Spre[s] ** 2).sum(0)

REAL = ["level", "runner", "trans", "echo"]
OTHER = ["h2", "alt", "h3", "short"]
Rc, Lc = RIGHT["clock"], LEFT["clock"]
results = {}
for b in BINS:
    Y = np.stack([num[b][a] / den[b][a] for a in AG])                  # (8, 13) orbit per unit M
    resid = np.stack([np.sqrt(np.maximum(sq[b][a] / den[b][a] - Y[i] ** 2, 0)) for i, a in enumerate(AG)])   # rms residual / rms M
    C = mode_parts(Y, REAL)
    K = mode_parts(Y, ["clock"])
    O = mode_parts(Y, OTHER)
    zc = Y @ Lc
    fo = np.stack([off[b][a] / cnt[b][a] for a in AG])
    fn = np.stack([neg[b][a] / cnt[b][a] for a in AG])
    results[b] = dict(Y=Y, C=C, K=K, O=O, zc=zc, fo=fo, fn=fn)
    print("\n" + "=" * 140)
    print(f"S/M = {b}  (bin {BINS[b][0]:.2f} < S+/M <= {BINS[b][1]:.2f});  states per age a=2..9: {[int(cnt[b][a]) for a in AG]}")
    print(f"   mode coordinates of the orbit / 10.157:  level {np.round(Y @ LEFT['level'].real / 10.157, 3)}")
    print(f"                                            runner {np.round(Y @ LEFT['runner'].real / 10.157, 2)}")
    print(f"                                            clock |z| {np.round(abs(zc) / 10.157, 3)}  phase {np.round(np.degrees(np.angle(zc)) % 360, 0)}")
    print("   orbit y_i(a) per unit M (rows: age a = 2..9; columns: units " + str(ORD) + ")")
    for i, a in enumerate(AG):
        print(f"     a={a} | " + " ".join(f"{v:7.2f}" for v in Y[i]))
    print("   per unit:   centre (mean over ages of the level+runner+trans+echo part) | clock radius 2|z||R_i| (mean over ages; at a=2) | radius/centre |"
          " min over ages of y_i (age) | measured: P(pre-input state <= 0) at that age, P(unit OFF) at that age | largest P(unit OFF) over ages (age)")
    for i, u in enumerate(ORD):
        cen = C[:, i].mean()
        rad = 2 * abs(zc) * abs(Rc[i])
        k = int(np.argmin(Y[:, i]))
        km = int(np.argmax(fo[:, i]))
        print(f"     unit {u:2d} ({'comp' if u in COMP else 'wave'}) | centre {cen:6.2f} | radius {rad.mean():6.2f} (a=2: {rad[0]:6.2f}) | {rad.mean() / cen:5.2f} |"
              f" min {Y[k, i]:6.2f} (a={AG[k]}) | {fn[k, i]:.3f}  {fo[k, i]:.3f} | {fo[km, i]:.3f} (a={AG[km]})")
    # --- centre removed
    Rot = Y - C
    negfrac = (Rot < 0).mean()
    lost = (np.minimum(Rot, 0) ** 2).sum() / (Rot ** 2).sum()
    ratio = (np.maximum(Rot, 0) @ Lc) / (Rot @ Lc)
    lev_new = np.maximum(Rot, 0) @ LEFT["level"].real / (Y @ LEFT["level"].real)
    print(f"   centre removed (linear model): share of (unit, age) entries of y - centre that are negative = {negfrac:.3f};"
          f" share of |y - centre|^2 in the negative entries = {lost:.3f}")
    print(f"      clock coordinate of ReLU(y - centre) relative to that of y - centre: |ratio| per age {np.round(abs(ratio), 2)}, phase (deg) {np.round(np.degrees(np.angle(ratio)), 0)}")
    print(f"      level coordinate of ReLU(y - centre) relative to the original level: {np.round(lev_new, 2)}")
    # with the real centre: how many entries negative
    print(f"   with the centre: share of (unit, age) entries of y that are negative = {(Y < 0).mean():.3f}: "
          + ", ".join(f"(a={AG[i]}, u{ORD[j]}: {Y[i, j]:.2f})" for i, j in zip(*np.where(Y < 0))))
    # --- plane geometry of the mean orbit (same recipe as the owner's impulse PCA: fit on a = 2..9, 13 ordinary units)
    mu = Y.mean(0)
    _, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    V2 = Vt[:2]
    nperp = mu - V2.T @ (V2 @ mu)
    P2 = (Y - mu) @ V2.T
    ang = np.degrees(np.unwrap(np.arctan2(P2[:, 1], P2[:, 0])))
    qc, _ = np.linalg.qr(np.stack([Rc.real, Rc.imag], 1))
    pa = np.degrees(np.arccos(np.clip(np.linalg.svd(qc.T @ V2.T, compute_uv=False), -1, 1)))
    lev = RIGHT["level"].real
    print(f"   plane geometry of the mean orbit (PCA over a = 2..9): PC shares {np.round(sv[:3] ** 2 / (sv ** 2).sum(), 4)}; |mean| = {np.linalg.norm(mu):.2f};"
          f" distance origin-plane |n| = {np.linalg.norm(nperp):.2f}; in-plane rms radius = {np.sqrt((P2 ** 2).sum(1).mean()):.2f};"
          f" off-plane rms = {np.sqrt((sv[2:] ** 2).sum() / len(AG)):.2f}")
    print(f"      rotation per step (deg) {np.round(np.abs(np.diff(ang)), 1)} mean {np.abs(np.diff(ang)).mean():.2f}; principal angles to the clock plane {np.round(pa, 1)};"
          f" cos(mean, level eigvec) = {mu @ lev / np.linalg.norm(mu) / np.linalg.norm(lev):.4f};"
          f" rms 13-dim norm of: centre part {np.sqrt((C ** 2).sum(1).mean()):.2f}, clock part {np.sqrt((K ** 2).sum(1).mean()):.2f}, other oscillating modes {np.sqrt((O ** 2).sum(1).mean()):.2f}")

# --- comparison with the D1-type linear model: s ~ (M - S+) u~(a) + S+ v~(a), fitted on all states with M > 0 ("pos" rule)
print("\n" + "=" * 140)
print("Linear model check: pre-input state from the regression on all M > 0 states, s ~ (M - S+) u(a) + S+ v(a) (regressors g, S; seeds 1001-1008),")
print("evaluated at S/M = 0, 0.5, 0.9 and compared with the binned orbits above (rms difference over the 13 units x 8 ages, and rms of the orbit)")
Gp = load_gram("pos").sum(0).sum(1)                   # (a1, K, K)
for b, sv in (("0", 0.0), ("0.5", 0.5), ("0.9", 0.9)):
    P = []
    for a in AG:
        f = fit(Gp[a], ["g", "S"], YZ)["B"]
        fx = fit(Gp[a], ["g", "S"], [3])["B"][:, 0]                    # E_lin[x_t | g, S]
        Bs = f - np.outer(fx, w[oi])
        P.append((1 - sv) * Bs[0] + sv * Bs[1])
    P = np.stack(P)
    Y = results[b]["Y"]
    print(f"   S/M = {b}: rms difference {np.sqrt(((P - Y) ** 2).mean()):.2f};  rms of the binned orbit {np.sqrt((Y ** 2).mean()):.2f};"
          f"  largest |difference| {abs(P - Y).max():.2f} (a={AG[np.unravel_index(abs(P - Y).argmax(), Y.shape)[0]]}, unit {ORD[np.unravel_index(abs(P - Y).argmax(), Y.shape)[1]]})")
